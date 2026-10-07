"""The MoE method must leave parameters it does not own to their own loaders.

vLLM's RoutedExperts assigns the router's ``e_score_correction_bias``
(DeepSeek-V3-style ``noaux_tc`` routing) to itself before it calls
``create_weights``, so the bias is already a parameter of the MoE layer when the
EXL3 method registers its own. ``create_weights`` used to install the EXL3
loader on every direct parameter of the layer, the bias included, and that
loader declines a tensor that arrives without MoE kwargs -- so the bias was
never loaded and routing ran without it. Nothing failed: vLLM does not report
unloaded parameters for quantized models.

CPU only: the layer is built on the host and no kernel runs.
"""
from types import SimpleNamespace

import torch

import cuda_exl3.moe as moe
from cuda_exl3.moe import Exl3MoEMethod, _exl3_moe_weight_loader

E, H, I, BITS, N_ROUTED = 4, 128, 128, 2, 16
OWN = {"w13_trellis", "w2_trellis", "w13_suh", "w2_suh", "w13_svh", "w2_svh",
       "w13_mcg", "w2_mcg"}


class _RoutedExperts(torch.nn.Module):
    """The MoE layer as create_weights receives it: the router bias is already
    one of its parameters."""

    def __init__(self, bias):
        super().__init__()
        self.e_score_correction_bias = bias


def _build(monkeypatch, bias):
    # create_weights allocates on the current CUDA device; keep it on the host.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    layer = _RoutedExperts(bias)
    info = SimpleNamespace(bits=BITS, cb=1)                     # cb 1 = mcg
    method = Exl3MoEMethod(SimpleNamespace(tp_size=1, tp_rank=0), None,
                           info, info, info, "model.layers.3.mlp.experts")
    # extra_weight_attrs carries vLLM's generic MoE loader; a stub stands in.
    method.create_weights(layer, num_experts=E, hidden_size=H,
                          intermediate_size_per_partition=I,
                          params_dtype=torch.bfloat16,
                          weight_loader=lambda *a, **k: None)
    return method, layer


def _bias():
    """The model's bias before loading (it allocates it with torch.empty)."""
    return torch.nn.Parameter(torch.zeros(N_ROUTED), requires_grad=False)


def _ckpt_bias():
    return torch.linspace(5.0, 9.0, N_ROUTED)


def _load_like_the_model(param, w):
    """How a model's load_weights loads a non-expert tensor (deepseek_v2, ...)."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    getattr(param, "weight_loader", default_weight_loader)(param, w)


def _warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(moe.logger, "warning",
                        lambda msg, *a, **k: seen.append(msg % a))
    return seen


def test_create_weights_leaves_the_router_bias_alone(monkeypatch):
    bias = _bias()
    _, layer = _build(monkeypatch, bias)
    assert layer.e_score_correction_bias is bias
    assert not hasattr(bias, "weight_loader")      # the model's default applies
    assert not hasattr(bias, "_exl3_method")


def test_a_bias_loader_the_model_installed_survives(monkeypatch):
    """Some models give the bias a loader of their own (MiniMax-M2)."""
    def model_loader(param, w):
        param.data.copy_(w.float())

    bias = _bias()
    bias.weight_loader = model_loader
    _build(monkeypatch, bias)
    assert bias.weight_loader is model_loader


def test_loading_writes_the_router_bias(monkeypatch):
    bias = _bias()
    _, layer = _build(monkeypatch, bias)
    want = _ckpt_bias()
    _load_like_the_model(layer.e_score_correction_bias, want)
    assert torch.equal(bias.data, want)


def test_own_parameters_keep_the_exl3_loader(monkeypatch):
    _, layer = _build(monkeypatch, _bias())
    got = {n for n, p in layer.named_parameters(recurse=False)
           if getattr(p, "weight_loader", None) is _exl3_moe_weight_loader}
    assert got == OWN
    # ...and it still places a per-expert tensor that arrives with MoE kwargs.
    w = torch.randn(H).half()
    assert _exl3_moe_weight_loader(layer.w2_svh, w, "experts.w2_svh",
                                   shard_id="w2", expert_id=1, return_success=True)
    assert torch.equal(layer.w2_svh.data[1], w)


def test_own_tensor_without_moe_kwargs_is_still_declined(monkeypatch):
    """Unchanged: a per-expert tensor cannot be placed without an expert id."""
    _, layer = _build(monkeypatch, _bias())
    before = layer.w2_svh.data.view(torch.int16).clone()   # bitwise: may hold NaN
    assert _exl3_moe_weight_loader(layer.w2_svh, torch.ones(E, H).half(),
                                   return_success=True) is False
    assert torch.equal(layer.w2_svh.data.view(torch.int16), before)


def test_a_foreign_tensor_reaching_the_exl3_loader_is_loaded_not_dropped():
    p = _bias()
    want = _ckpt_bias()
    assert _exl3_moe_weight_loader(p, want, return_success=True) is True
    assert torch.equal(p.data, want)


def test_an_unloaded_router_bias_is_warned_about(monkeypatch):
    method, layer = _build(monkeypatch, _bias())
    seen = _warnings(monkeypatch)
    method.process_weights_after_loading(layer)
    assert any("e_score_correction_bias" in s for s in seen), seen


def test_a_loaded_router_bias_is_not_warned_about(monkeypatch):
    method, layer = _build(monkeypatch, _bias())
    _load_like_the_model(layer.e_score_correction_bias, _ckpt_bias())
    seen = _warnings(monkeypatch)
    method.process_weights_after_loading(layer)
    assert seen == []


def test_a_layer_without_a_router_bias(monkeypatch):
    method, layer = _build(monkeypatch, None)
    seen = _warnings(monkeypatch)
    method.process_weights_after_loading(layer)
    assert layer.e_score_correction_bias is None and seen == []
