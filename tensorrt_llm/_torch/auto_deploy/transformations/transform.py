"""High-level entrypoint to transform a model into an efficient inference model."""

import gc

import torch
from torch.fx import GraphModule

from ....llmapi.llm_args import _AutoDeployLlmArgs
from ..compile import compile_and_capture
from ..custom_ops.attention_interface import AttentionRegistry
from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
from ._graph import canonicalize_graph, lift_to_meta, move_to_device
from .export import torch_export_to_gm
from .library import (
    column_row_shard,
    dp_bmm_shard,
    eliminate_redundant_transposes,
    ep_shard,
    fuse_allreduce_residual_rmsnorm,
    fuse_collectives,
    insert_cached_attention,
    match_attention_layout,
    match_causal_attn_mask,
    match_eager_attention,
    match_grouped_attention,
    match_moe_pattern,
    match_repeat_kv,
    match_rms_norm,
    match_rope_layout,
    match_rope_pattern,
    optimize_rope,
    quantize,
    resize_kv_cache,
    update_in_out_nodes,
)


class InferenceOptimizer:
    def __init__(
        self,
        factory: ModelFactory,
        *,  # TODO: temporary until we have a better config system
        ad_config: _AutoDeployLlmArgs,
        visualize: bool = False,
    ):
        self.factory = factory

        self.ad_config = ad_config
        # Map Pytorch config to AutoDeploy compile backends.
        if ad_config.use_cuda_graph and ad_config.torch_compile_enabled:
            compile_backend = "torch-opt"
        elif ad_config.use_cuda_graph:
            compile_backend = "torch-cudagraph"
        elif ad_config.torch_compile_enabled:
            compile_backend = "torch-compile"
        else:
            compile_backend = "torch-simple"
        self.compile_backend = compile_backend
        self.visualize = visualize

    def __call__(self, cm: CachedSequenceInterface) -> GraphModule:
        """Transform a model into an optimized inference model.

        Args:
            model: The model to transform.
            cp: The cache pool to use for caching.
            args: Example inputs to the model.
            dynamic_shapes: Dynamic shapes to use. Defaults to None.
            poe_config: The config for positional encoding. Defaults to None.
            quantization: The quantization method to use. Defaults to None.

        Returns:
            A GraphModule representing the optimized inference model.
        """
        ############################################################################################
        # INITIALIZE MODEL
        ############################################################################################
        model = self.factory.build_model(device="meta")

        ############################################################################################
        # EXPORT MODEL TO GRAPH MODULE
        ############################################################################################

        cm.info.set_example_sequence()
        egm = torch_export_to_gm(model, args=cm.args, dynamic_shapes=cm.dynamic_shapes)
        del model
        ad_logger.debug("original graph: " + str(egm))
        local_rank, world_size = dist_ad.get_rank_world_size()

        ############################################################################################
        # RUN PATTERN MATCHER TRANSFORMATIONS TO STANDARDIZE GRAPH REPRESENTATION
        ############################################################################################

        # quantization
        egm = quantize(egm, self.factory.get_quant_config())

        # Match MoE pattern
        egm = match_moe_pattern(egm)

        # Match repeat_kv pattern
        egm = match_repeat_kv(egm)

        # Match eager attention pattern
        egm = match_eager_attention(egm)

        # Match grouped attention pattern
        egm = match_grouped_attention(egm)

        # Match and optimize causal attention masks
        egm = match_causal_attn_mask(egm)

        # Match attention layout expected by our backend
        egm = match_attention_layout(egm, AttentionRegistry.get(self.ad_config.attn_backend))

        # Match rope
        egm, _ = match_rope_pattern(egm)

        # Match RoPE layout expected by our backend
        egm = match_rope_layout(
            egm, AttentionRegistry.get(self.ad_config.attn_backend).get_attention_layout()
        )

        ############################################################################################
        # RUN TRANSFORMATIONS ON STANDARDIZED GRAPH REPRESENTATION
        ############################################################################################

        # eliminate redundant transpose operations
        egm = eliminate_redundant_transposes(egm)

        # TODO (lucaslie): let's move this to perf optimization once TP sharding is improved
        # see https://github.com/NVIDIA/TensorRT-LLM/pull/3668#discussion_r2052714528
        egm = optimize_rope(egm)

        # run TP sharding across ranks
        egm = column_row_shard(egm, local_rank, world_size, self.ad_config.simple_shard_only)

        # run EP sharding across ranks
        egm = ep_shard(egm, local_rank, world_size)

        # run BMM sharding across ranks
        egm = dp_bmm_shard(egm, local_rank, world_size)

        # let's run a shape propagation pass to update the graph with correct meta values for
        # subsequent optimization passes. Lift state_dict to meta as shape propagation involves device check
        with lift_to_meta(egm):
            egm = canonicalize_graph(egm, shape_prop=True)

        ############################################################################################
        # MOVE MODEL AND LOAD WEIGHTS
        ############################################################################################

        # load weights
        self.factory.load_or_random_init(egm, device=self.ad_config.checkpoint_device or cm.device)

        # move remaining parts to device
        move_to_device(egm, cm.device)
        cm.to(cm.device)

        ############################################################################################
        # RUN POST-LOAD FUSION AND OPTIMIZATIONS
        ############################################################################################

        # run MoE fusion
        # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 this is causing OOMs
        # egm = fuse_moe(egm)

        # run GEMM fusion
        # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 this is causing OOMs
        # egm = fuse_gemms(egm)

        # check if we can fuse allreduce, residual and rmsnorm
        egm = fuse_allreduce_residual_rmsnorm(egm)

        # check if we can fuse collectives
        egm = fuse_collectives(egm)

        # match rms norm pattern
        egm = match_rms_norm(egm)

        # visualize the final graph
        if self.visualize:
            try:
                from .library import visualize_namespace

                visualize_namespace(egm, args=cm.args, dynamic_shapes=cm.dynamic_shapes)
                ad_logger.warning(
                    "Please run `pip install -r examples/auto_deploy/requirements.txt` to visualize"
                    " the graph."
                )
            except ImportError:
                pass

        ############################################################################################
        # SWITCH TO CACHED+FLATTENED ATTENTION + INITIALIZE CACHES
        ############################################################################################

        egm = update_in_out_nodes(egm, cm)

        # detect attention op and replace with cache-aware op
        for a_backend in [self.ad_config.attn_backend, self.ad_config.mla_backend]:
            attn_descriptor = AttentionRegistry.get(a_backend)
            egm = insert_cached_attention(egm, cm, attn_descriptor, self.factory.get_cache_config())

        # initialize cache on correct device
        cm.initialize_caches()

        # resize kv cache to occupy the available GPU memory up to free_mem_ratio
        resize_kv_cache(egm, cm, free_mem_ratio=self.ad_config.free_mem_ratio)

        ############################################################################################
        # COMPILE MODEL
        ############################################################################################

        cm.info.set_generate_only_batch()
        compiler_kwargs = {
            "cuda_graph_batch_sizes": self.ad_config.cuda_graph_batch_sizes,
            "num_batched_inputs": 2,  # TODO (lucaslie): improve once we have a config system...
        }
        egm_compiled = compile_and_capture(
            egm,
            self.compile_backend,
            args=cm.args,
            dynamic_shapes=cm.dynamic_shapes,
            compiler_kwargs=compiler_kwargs,
        )
        cm.info.reset()

        torch.cuda.empty_cache()
        gc.collect()
        return egm_compiled


class ThunderInferenceOptimizer:
    def __init__(
        self,
        factory: ModelFactory,
        *,  # TODO: temporary until we have a better config system
        ad_config: _AutoDeployLlmArgs,
        visualize: bool = False,
    ):
        self.factory = factory

        self.ad_config = ad_config
        # Map Pytorch config to AutoDeploy compile backends.
        if ad_config.use_cuda_graph and ad_config.torch_compile_enabled:
            compile_backend = "torch-opt"
        elif ad_config.use_cuda_graph:
            compile_backend = "torch-cudagraph"
        elif ad_config.torch_compile_enabled:
            compile_backend = "torch-compile"
        else:
            compile_backend = "torch-simple"
        self.compile_backend = compile_backend
        self.visualize = visualize

    def __call__(self, cm: CachedSequenceInterface) -> GraphModule:
        """Transform a model into an optimized inference model.

        Args:
            model: The model to transform.
            cp: The cache pool to use for caching.
            args: Example inputs to the model.
            dynamic_shapes: Dynamic shapes to use. Defaults to None.
            poe_config: The config for positional encoding. Defaults to None.
            quantization: The quantization method to use. Defaults to None.

        Returns:
            A GraphModule representing the optimized inference model.
        """
        ############################################################################################
        # INITIALIZE MODEL
        ############################################################################################
        model = self.factory.build_model(device="meta")

        ############################################################################################
        # EXPORT MODEL TO GRAPH MODULE
        ############################################################################################

        cm.info.set_example_sequence()
        local_rank, world_size = dist_ad.get_rank_world_size()

        ############################################################################################
        # MOVE MODEL AND LOAD WEIGHTS
        ############################################################################################

        # load weights
        self.factory.load_or_random_init(model, device=self.ad_config.checkpoint_device or cm.device)

        # move remaining parts to device
        move_to_device(model, cm.device)
        cm.to(cm.device)
        import copy
        self.model = None

        ############################################################################################
        # COMPILE MODEL
        ############################################################################################

        #cm.info.set_generate_only_batch()
        #compiler_kwargs = {
        #    "cuda_graph_batch_sizes": self.ad_config.cuda_graph_batch_sizes,
        #    "num_batched_inputs": 2,  # TODO (lucaslie): improve once we have a config system...
        #}

        from thunder.dynamo import thunder_profile, thunder_optimize
        pmodel = thunder_profile(model)
        print(len(cm.args),cm.args[0].shape, cm.args[1].shape)
        #pmodel(*cm.args)
        a_list = []
        b_list= []
        for shape in [[1, 128], [1, 129], [2, 1] ,[1, 1], [1,143]]:
            a = torch.ones(shape, dtype=torch.int32, device="cuda")
            b = torch.ones(shape, dtype=torch.int64, device="cuda")
            pmodel(a,b)
            a_list.append(a)
            b_list.append(b)
        print("len of pmodel._tao.id_to_profile_stats: ",len(pmodel._tao.id_to_profile_stats))
        new_args = cm.info.switch_to_cached_attn_inputs()
        attn_descriptor = AttentionRegistry.get(self.ad_config.attn_backend)
        cache_config = self.factory.get_cache_config()
        from thunder.dynamo.utils import KVCacheManager as thunder_cache_manager
        cmanager = thunder_cache_manager(cm, attn_descriptor, cache_config, self.ad_config)


        def optim(gm, stats):
            #print(gm)
            #gm1 = cmanager.transform_graph(stats.gm,cm,new_args)
            #print(gm1)
            gm1=gm
            from thunder.dynamo.utils import has_symbolic_input
            #if not has_symbolic_input(gm1):
            #    return gm1
            placeholders = [n for n in stats.gm.graph.nodes if n.op == "placeholder"]
            #example_inputs_meta = [_get_example_inputs_from_placeholder(p, only_metadata=True) for p in placeholders]
            from thunder.dynamo.utils import get_or_create_example_inputs_from_placeholders
            example_inputs = get_or_create_example_inputs_from_placeholders(placeholders)
            
            from ..compile.backends.thunder_compiler1 import ThunderOptCompiler 
            cm.info.set_generate_only_batch()
            compiler_kwargs = {
                "cuda_graph_batch_sizes": self.ad_config.cuda_graph_batch_sizes,
                "num_batched_inputs": 2,  # TODO (lucaslie): improve once we have a config system...
            }
            compiler_kwargs["cuda_graph_batch_sizes"]=[1, 128, 256, 384, 512]#[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 256, 512] #[1, 128, 256, 384, 512]#, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792]
            return ThunderOptCompiler(gm1,example_inputs,cm,dynamic_shapes = cm.dynamic_shapes,compiler_kwargs=compiler_kwargs).compile()

            #gm1=gm

            #print(id(model))
            #from thunder.dynamo.benchmark_utils import ThunderCompilerOnGraphModuleSpecification
            #import thunder
            #thunder_compiler_on_gm = ThunderCompilerOnGraphModuleSpecification(nv_skip_cache=False,)
            #split_gm, bd = thunder_compiler_on_gm.compile(gm1)
            #from thunder.dynamo.utils import _readable
            #with open("/home/wayan/trtllm/dispatch_thunder_gms/split_gm.py",'w') as f:
            #    f.write(str(_readable(split_gm,"gmodule")))
            ##bd.save_reproducer_to_folder("/home/wayan/trtllm/dispatch_thunder_gms")
            #return split_gm
            #cm.info.pages_per_seq.fill_(0)
            #gm1(*example_inputs)
            #print("end exampleinput++++++++++++++++")
            #exit()

            #from thunder.dynamo.benchmark_utils import TorchInductorSpecification
            #torchinductor = TorchInductorSpecification()
            #return torchinductor.compile(gm1, inputs=example_inputs)

            #return torch.compile(gm)
            #return gm

        print(len(cm.args))  #2

        from thunder.dynamo.utils import default_filter
        from functools import partial
        import thunder
        egm_compiled, prof_model = thunder_optimize(pmodel, gm_filter=partial(default_filter, cutoff=1), optimizer=optim)

        cm.info.reset()

        torch.cuda.empty_cache()
        gc.collect()
        #return egm_compiled, egm_compiled
        return egm_compiled, self.model
