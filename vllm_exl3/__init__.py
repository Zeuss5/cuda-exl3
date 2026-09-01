"""EXL3 (ExLlamaV3 trellis quantization) support for vLLM.

vLLM discovers this package through the ``vllm.general_plugins`` entry point and
calls :func:`register`, which makes ``quant_method: "exl3"`` checkpoints loadable.
"""

__version__ = "0.1.0"

_REGISTERED = False


def register() -> None:
    """Register the EXL3 quantization method with vLLM.

    Called by vLLM's plugin loader, potentially once per process (engine core,
    each worker), so it must be idempotent.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # Importing the module runs the @register_quantization_config decorator.
    from vllm_exl3 import config as _config  # noqa: F401

    _REGISTERED = True


__all__ = ["register", "__version__"]
