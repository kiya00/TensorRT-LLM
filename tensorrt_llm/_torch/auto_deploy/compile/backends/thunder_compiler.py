from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.utils._pytree import TreeSpec, tree_flatten

from ...utils.cuda_graph import CudaGraphWarmUpPhase
from ...utils.logger import ad_logger
from ..compiler import BackendCompiler, BackendRegistry, _flatten_args

# Since the Dynamo graph treats all parameters as graph inputs, we identify actual input tensors by checking the type of each placeholder node.
def _get_input_tensors(gm, args,):
    node_inputs = gm.graph.find_nodes(op="placeholder", sort=True)
    input_tensor_idx_val_list = [[idx, arg] for idx,(n,arg) in enumerate(zip(node_inputs,args)) if n.type is torch.Tensor and not n.type is torch.nn.Parameter]
    return input_tensor_idx_val_list


def replace_input(target, indexes, result):
    assert len(target)==len(indexes), "replace inputs mismatch"
    t_iter = iter(target)
    return [next(t_iter) if idx in indexes else a for idx,a in enumerate(result)]


def pad_to_bucket(input_ids, position_ids, bucket_bs, pad_token_id=0):
    bs = input_ids.shape[0]
    pad_len = bucket_bs - bs
    assert pad_len >= 0, "bucket_bs must be >= bs"

    if pad_len > 0:
        pad_input_ids = torch.full((pad_len, 1), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        pad_position_ids = torch.zeros((pad_len, 1), dtype=position_ids.dtype, device=position_ids.device)

        input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
        position_ids = torch.cat([position_ids, pad_position_ids], dim=0)

    return input_ids, position_ids, bs
class CapturedGraph(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        cm,
        max_batch_size: int,
        cuda_graph_batch_sizes: List[int] = None,
        num_batched_inputs: Optional[int] = 1,  # number of batched, dynamic inputs...
    ):
        super().__init__()
        self.model = model
        self.cm = cm
        self.max_batch_size = max_batch_size
        self.num_batched_inputs = num_batched_inputs if num_batched_inputs is not None else 1
        self.graphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._input_buffers: List[torch.Tensor] = [
            torch.empty(0, 1) for _ in range(self.num_batched_inputs)
        ]
        self._out_buffer_flat: List[torch.Tensor] = None
        self._args_hash: Optional[Tuple[int, ...]] = None
        self.cuda_graph_batch_sizes = (
            cuda_graph_batch_sizes
            if cuda_graph_batch_sizes is not None
            else self._get_graph_batch_sizes(self.max_batch_size)
        )

    def _get_hash(self, flat_args: List[Any]) -> Tuple[int, ...]:
        return tuple(hash(a) for a in flat_args)

    @staticmethod
    def round_up_to_closest(batch_sizes: Iterable[int], bs: int) -> Optional[int]:
        """Return closest batch size larger or equal to bs."""
        if bs > max(batch_sizes, default=0):
            return None
        return min(batch_sizes, key=lambda x: (x < bs, abs(x - bs)), default=None)

    def round_to_cuda_batch_size(self, bs: int) -> int:
        """Round batch size to the nearest cuda batch size."""
        return self.round_up_to_closest(self.cuda_graph_batch_sizes, bs)

    @staticmethod
    def _get_graph_batch_sizes(
        max_bs: int, extra: Optional[List[int]] = None, multiplier: int = 128
    ) -> List[int]:
        """Heuristic to set batch sizes for graph capture."""
        # do 1, max_bs, and extra as special batch sizes
        batch_sizes = {1, max_bs, *(extra or [])}

        # add all multiples of multiplier up to max_bs
        batch_sizes.update(range(multiplier, max_bs + 1, multiplier))

        # return as sorted list
        return sorted(batch_sizes)

    @staticmethod
    def map_to_new_args(old_args, args_idxes, new_batched_args, bs):
        new_args_iter=iter(new_batched_args)
        all_new_args = [next(new_args_iter) if idx in args_idxes else arg for idx,arg in enumerate(old_args)]

        # replace the dynamic shape symInt with the current used shape
        for idx,a in enumerate(all_new_args):
            if isinstance(a, int):
                all_new_args[idx]=bs
        return all_new_args


    def capture_graph(self, *args, **kwargs):
        """Capture and pre-fetch the graph for variable batch size."""
        # flatten args, kwargs

        # extract the batched input tensors
        # args correspond to the dynamo graph inputs(including parameters), cm.args only contains original inputs
        args_lists = _get_input_tensors(self.model, args)[:self.num_batched_inputs]
        args_idxes = [a[0] for a in args_lists]
        args_batched = self.cm.args[:self.num_batched_inputs]
        # TODO: there is problem when using max_batch_size=2048, seems oom
        self.max_batch_size=512

        # set the args hash --> this is used to compare the static inputs during graph replay
        # self._args_hash = self._get_hash(args_static)

        # sanity checks on the batched inputs
        msg_bs = "Max batch size too small."
        msg_ndim = "Expecting at least a 2D for batched input tensors."
        # assert all(self.max_batch_size >= input.shape[0] for input in args_batched), msg_bs
        assert all(input.ndim > 1 for input in args_batched), msg_ndim

        self._input_buffers = [
            input[:1].repeat_interleave(self.max_batch_size, dim=0) for input in args_batched
        ]
        new_args = self.map_to_new_args(args, args_idxes, self._input_buffers, self.max_batch_size)
        # capture output once with max batch size to capture output buffers
        with CudaGraphWarmUpPhase():
            out = self.model(*new_args)
        self._out_buffer_flat, self.out_spec = tree_flatten(out)

        # capture graph now for a range of batch sizes
        for bs in self.cuda_graph_batch_sizes:
            ad_logger.info(f"Capturing graph for batch size: {bs}, {self.cuda_graph_batch_sizes}")

            inputs_truncated = [in_buffer[:bs] for in_buffer in self._input_buffers]

            new_args = self.map_to_new_args(args, args_idxes, inputs_truncated, bs)

            # capture graph for truncated inputs
            combined_shape = sum((input.shape for input in inputs_truncated), start=())

            # first apply thunder
            from thunder.dynamo import ThunderCompiler
            from thunder.executors.custom_op_ex import custom_op_ex
            from thunder import get_default_executors
            executor_list = get_default_executors()
            thunder_compiler = ThunderCompiler(executors=[*executor_list, custom_op_ex])
            split_gm = thunder_compiler(self.model, sample_args=None)
            #split_gm = torch.compile(self.model)
            #split_gm = self.model

            # then apply cudagraph
            with CudaGraphWarmUpPhase():
                for _ in range(3):
                    split_gm(*new_args)

            # capture graph now
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # compute output
                out = split_gm(*new_args)
                # write out into output buffer up to out batch size
                out_flat, out_spec = tree_flatten(out)
                assert out_spec == self.out_spec, "Output spec mismatch."
                for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                    o_buffer[: o.shape[0]] = o
            torch.cuda.synchronize()

            self.graphs[combined_shape] = graph

    def forward(self, *args, **kwargs) -> Any:
        """Run the compiled graph."""
        # extract the batched input tensors
        args_lists = _get_input_tensors(self.model, args)[:self.num_batched_inputs]
        args_idxes = [a[0] for a in args_lists]
        args_batched = [args[i] for i in args_idxes]
        args_batched = args_batched[:self.num_batched_inputs]
        ori_bs=args_batched[0].shape[0]

        # check if args_static match the stored hash
        # if self._args_hash != self._get_hash(args_static):
        #     return self.model(*args, **kwargs)

        # Calculate rounded-up shapes for each input
        rounded_shapes = [
            (self.round_to_cuda_batch_size(input.shape[0]),) + input.shape[1:]
            for input in args_batched
        ]
        combined_shape = sum(rounded_shapes, start=())

        # regular forward for non-matching shapes
        if combined_shape not in self.graphs:
            ad_logger.debug(f"bucket fallback to eager: {args_batched[0].shape}, {args_batched[1].shape}")
            return self.model(*args)
        ad_logger.debug(f"bucket hit: {args_batched[0].shape}, {args_batched[1].shape}, round to: {rounded_shapes}")

        # copy inputs to input buffers
        for i, input_tensor in enumerate(args_batched):
            self._input_buffers[i][: input_tensor.shape[0]] = input_tensor

        # run forward pass via graph
        self.graphs[combined_shape].replay()

        # retrieve output from buffer, cut to batch size, and unflatten
        out=[o[:ori_bs].detach().clone() for o in self._out_buffer_flat]
        return self.out_spec.unflatten(out)

@BackendRegistry.register("thunder")
class ThunderOptCompiler(BackendCompiler):
    def __init__(
        self,
        gm,
        args: Tuple[Any, ...],
        cm,
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes=None,
        compiler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.gm = gm
        self.args = args
        self.cm = cm
        self.kwargs = kwargs or {}
        self.dynamic_shapes = dynamic_shapes
        self.compiler_kwargs = compiler_kwargs or {}
        # identify max_batch_size
        if self.dynamic_shapes is not None and 0 in self.dynamic_shapes[0]:
            self.max_batch_size = self.dynamic_shapes[0][0].max
        else:
            # NOTE: we assume the first input is the main input tensor with batch dimension
            #batched_input, *_ = _flatten_args(self.gm._in_spec, *self.args, **self.kwargs)
            # self.max_batch_size = batched_input.shape[0]
            input_tensor_idx_val_list = _get_input_tensors(self.gm, self.args)
            self.max_batch_size = input_tensor_idx_val_list[0][1].shape[0]


    def _init_captured_graph(
        self, gm: nn.Module
    ) -> CapturedGraph:
        return CapturedGraph(
            gm,
            self.cm,
            max_batch_size=self.max_batch_size,
            cuda_graph_batch_sizes=self.compiler_kwargs.get("cuda_graph_batch_sizes"),
            num_batched_inputs=self.compiler_kwargs.get("num_batched_inputs"),
        )

    @torch.inference_mode()
    def compile(self) -> CapturedGraph:
        fst_input = _get_input_tensors(self.gm, self.args)[0][1]
        # Bucketing is currently applied only along the batch size dimension.
        # If the sequence length is dynamic, the compiler falls back to eager mode.
        if fst_input.shape[0]==1: #1, total_len
            ad_logger.debug(f"fallback to eager for [1, total_len]: {fst_input.shape}")
            return self.gm

        captured_model = self._init_captured_graph(self.gm)

        # try capturing cudagraph
        if self.args is not None or self.kwargs is not None:
            captured_model.capture_graph(*self.args, **self.kwargs)

        return captured_model

