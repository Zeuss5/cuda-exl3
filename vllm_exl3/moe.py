"""EXL3 mixture-of-experts method for vLLM.

Routed experts are executed as a grouped GEMM. vLLM's `moe_align_block_size`
sorts the (token, expert) pairs by expert and pads each expert's run out to a
whole block, so every row block belongs to exactly one expert -- which is what
lets the GEMM offset the trellis by a single expert id per block rather than per
row.

The EXL3 wrinkle is that `suh` is per expert *and* per shard, and it sits inside
the input Hadamard, so the activation transform has to happen after routing:
each routed row is transformed with its own expert's scales. That is what
`exl3_moe_had_in` does, gathering and transforming in one pass.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

TILE = 16


class Exl3MoEMethod(FusedMoEMethodBase):
    """Routed experts with EXL3-quantized w13 / w2."""

    def __init__(self, moe, quant_config, gate_info, up_info, down_info, prefix: str):
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix

        bits = {gate_info.bits, up_info.bits}
        if len(bits) != 1:
            raise ValueError(
                f"EXL3 {prefix}: gate and up are quantized at different bitrates "
                f"{sorted(bits)}; they share one fused trellis and cannot differ."
            )
        self.w13_bits = bits.pop()
        self.w2_bits = down_info.bits
        cbs = {gate_info.cb, up_info.cb, down_info.cb}
        if len(cbs) != 1:
            raise ValueError(f"EXL3 {prefix}: experts use different codebooks")
        self.cb = cbs.pop()
        self.cb_name = {1: "mcg", 2: "mul1"}.get(self.cb)

    # -- weights ---------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        E = num_experts
        H = hidden_size
        I = intermediate_size_per_partition
        dev = torch.cuda.current_device()

        for name, dims in [
            ("w13_trellis", (E, H // TILE, 2 * I // TILE, TILE * self.w13_bits)),
            ("w2_trellis", (E, I // TILE, H // TILE, TILE * self.w2_bits)),
        ]:
            p = torch.nn.Parameter(
                torch.empty(dims, dtype=torch.int16, device=dev), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, {**extra_weight_attrs})

        for name, dims in [
            ("w13_suh", (E, 2, H)),      # gate and up have separate input scales
            ("w2_suh", (E, 1, I)),
            ("w13_svh", (E, 2 * I)),
            ("w2_svh", (E, H)),
        ]:
            p = torch.nn.Parameter(
                torch.empty(dims, dtype=torch.half, device=dev), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, {**extra_weight_attrs})

        # The codebook multiplier is a compile-time constant of the codebook id,
        # but it is still a tensor in the checkpoint, so it needs somewhere to go.
        if self.cb_name:
            for name, dims in [(f"w13_{self.cb_name}", (E, 2)),
                               (f"w2_{self.cb_name}", (E, 1))]:
                p = torch.nn.Parameter(
                    torch.empty(dims, dtype=torch.int32, device=dev), requires_grad=False
                )
                layer.register_parameter(name, p)
                set_weight_attrs(p, {**extra_weight_attrs})

        for pname, p in list(layer.named_parameters(recurse=False)):
            p._exl3_name = pname
        self._install_loader(layer)

        layer.exl3_num_experts = E
        layer.exl3_hidden = H
        layer.exl3_inter = I
        layer.exl3_cb = self.cb
        layer.exl3_w13_bits = self.w13_bits
        layer.exl3_w2_bits = self.w2_bits

    def _install_loader(self, layer):
        """Take over weight loading for this module.

        vLLM's MoE loader decides `is_fused = loaded_weight.dim() == 3`, i.e. it
        reads a 3-D tensor as "all experts stacked". An EXL3 trellis is
        genuinely 3-D *per expert* (k/16, n/16, 16*bits), so that heuristic
        slices it apart along the wrong axis. There is no hook before that
        decision, so this module loads its own weights.
        """
        method = self

        def load_weights(weights):
            loaded: set[str] = set()
            for name, w in weights:
                parts = name.split(".")
                if len(parts) < 3 or not parts[0].isdigit():
                    continue                      # not a per-expert tensor
                expert_id, proj, suffix = int(parts[0]), parts[1], parts[-1]
                local = method._local_expert(layer, expert_id)
                if local < 0:
                    continue                      # expert lives on another rank
                pname = method._place(layer, local, proj, suffix, w)
                if pname:
                    loaded.add(pname)
            return loaded

        layer.load_weights = load_weights

    @staticmethod
    def _local_expert(layer, expert_id: int) -> int:
        emap = getattr(layer, "expert_map", None)
        if emap is None:
            return expert_id
        return int(emap[expert_id].item()) if expert_id < emap.numel() else -1

    def _place(self, layer, e: int, proj: str, suffix: str, w: torch.Tensor):
        """Copy one checkpoint tensor into the fused per-expert parameter.

        gate/up occupy the two halves of w13 along the output dim, matching how
        vLLM lays out w13 everywhere else. Tensor-parallel slicing is on the
        intermediate dim: the output dim for gate/up, the input dim for down.
        """
        r = self._tp_rank
        if proj == "down_proj":
            if suffix == "trellis":
                p = layer.w2_trellis
                part = p.shape[1]
                p.data[e].copy_(w.narrow(0, r * part, part))
            elif suffix == "suh":
                p = layer.w2_suh
                part = p.shape[2]
                p.data[e][0].copy_(w.narrow(0, r * part, part))
            elif suffix == "svh":
                p = layer.w2_svh
                p.data[e].copy_(w)
            else:
                p = getattr(layer, f"w2_{self.cb_name}", None)
                if p is None:
                    return None
                p.data[e][0] = int(w.reshape(-1)[0])
            return p._exl3_name

        half = 0 if proj == "gate_proj" else 1
        if suffix == "trellis":
            p = layer.w13_trellis
            part = p.shape[2] // 2
            p.data[e].narrow(1, half * part, part).copy_(w.narrow(1, r * part, part))
        elif suffix == "suh":
            p = layer.w13_suh
            p.data[e][half].copy_(w)
        elif suffix == "svh":
            p = layer.w13_svh
            part = p.shape[1] // 2
            p.data[e].narrow(0, half * part, part).copy_(w.narrow(0, r * part, part))
        else:
            p = getattr(layer, f"w13_{self.cb_name}", None)
            if p is None:
                return None
            p.data[e][half] = int(w.reshape(-1)[0])
        return p._exl3_name

    @property
    def _tp_rank(self) -> int:
        return int(getattr(self.moe, "tp_rank", 0) or 0)

    def get_fused_moe_quant_config(self, layer):
        # Not using vLLM's modular kernel framework; apply() runs the whole thing.
        return None

    # -- execution -------------------------------------------------------

    @staticmethod
    def _block_m(rows: int, num_experts: int) -> int:
        """Row-block size, which is also the alignment granularity.

        Must be one of the GEMM's BM tiers. Larger blocks amortize the weight
        read over more rows, but each expert's run is padded up to a whole block,
        so with few tokens per expert the padding dominates.
        """
        per_expert = rows / max(num_experts, 1)
        if per_expert < 32:
            return 32
        return 64 if per_expert < 96 else 128

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        if shared_experts is not None:
            # This method is not a modular kernel, so it cannot overlap the
            # shared expert; SharedExperts.forward no-ops unless the order it
            # picked matches. Call it defensively (as the modular path does) and
            # return only the routed output -- the caller collects the shared
            # result from the SharedExperts object itself.
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)

        M, H = x.shape
        T = topk_ids.shape[1]
        E = layer.exl3_num_experts
        I = layer.exl3_inter
        out_dtype = x.dtype

        block_m = self._block_m(M * T, E)
        # pad_sorted_ids makes sorted_ids a whole number of blocks; without it
        # expert_ids covers more blocks than sorted_ids has entries and the
        # gather reads past the end.
        sorted_ids, expert_ids, n_rows = moe_align_block_size(
            topk_ids, block_m, E, pad_sorted_ids=True
        )
        sorted_ids = sorted_ids.int()
        expert_ids = expert_ids.int()
        n_rows = n_rows.int()
        rows = min(expert_ids.numel() * block_m, sorted_ids.numel())
        expert_ids = expert_ids[: rows // block_m]

        xc = x.contiguous()
        ops = torch.ops.vllm_exl3_C

        # gate/up: two transforms per routed row, one per shard's suh
        a13 = torch.empty((2, rows, H), dtype=torch.half, device=x.device)
        ops.exl3_moe_had_in(xc, a13, layer.w13_suh.data, sorted_ids, expert_ids,
                            n_rows, block_m, T, M * T)
        inter = ops.exl3_moe_gemm(a13, layer.w13_trellis.data, layer.w13_suh.data,
                                  layer.w13_svh.data, expert_ids, n_rows, [I, I],
                                  layer.exl3_cb, block_m, out_dtype)

        act = torch.nn.functional.silu(inter[:, :I]) * inter[:, I:]

        # down: one transform, this time of the activation
        a2 = torch.empty((1, rows, I), dtype=torch.half, device=x.device)
        ops.exl3_moe_had_in(act.contiguous(), a2, layer.w2_suh.data,
                            torch.arange(rows, device=x.device, dtype=torch.int32),
                            expert_ids, n_rows, block_m, 1, rows)
        rows_out = ops.exl3_moe_gemm(a2, layer.w2_trellis.data, layer.w2_suh.data,
                                     layer.w2_svh.data, expert_ids, n_rows, [H],
                                     layer.exl3_cb, block_m, out_dtype)

        # Scatter back to (token, k) order and combine. Padded rows are sent to a
        # scratch slot rather than masked, so nothing here needs a device sync
        # (which would break CUDA graph capture).
        buf = torch.zeros((M * T + 1, H), dtype=out_dtype, device=x.device)
        buf.index_copy_(0, sorted_ids.clamp(max=M * T).long(), rows_out)
        return (buf[: M * T].view(M, T, H) * topk_weights.to(out_dtype).unsqueeze(-1)).sum(1)
