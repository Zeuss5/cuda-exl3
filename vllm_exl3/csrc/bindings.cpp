#include <torch/extension.h>
#include <torch/library.h>

namespace vllm_exl3 {
void exl3_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh);
void exl3_reserve(const at::Tensor& like, int64_t max_tokens, int64_t k, int64_t n,
                  int64_t groups);
at::Tensor exl3_linear(const at::Tensor& x, const at::Tensor& trellis,
                       const at::Tensor& suh, const at::Tensor& svh,
                       at::IntArrayRef group_n, int64_t cb, bool split_k);
}  // namespace vllm_exl3

// Registered through torch.library rather than raw pybind so that torch.compile
// / Dynamo can trace the call (a bare pybind function is opaque to it, which
// blocks vLLM's CUDA graph capture). The op is functional -- it allocates and
// returns its output -- which also keeps it easy to give a meta implementation.
TORCH_LIBRARY(vllm_exl3_C, m)
{
    m.def(
        "exl3_linear(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "int[] group_n, int cb, bool split_k) -> Tensor");
    m.def("exl3_had_in(Tensor x, Tensor(a!) out, Tensor suh) -> ()");
    m.def("exl3_reserve(Tensor like, int max_tokens, int k, int n, int groups) -> ()");
}

TORCH_LIBRARY_IMPL(vllm_exl3_C, CUDA, m)
{
    m.impl("exl3_linear", &vllm_exl3::exl3_linear);
    m.impl("exl3_had_in", &vllm_exl3::exl3_had_in);
    m.impl("exl3_reserve", &vllm_exl3::exl3_reserve);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "EXL3 kernels; ops are registered under torch.ops.vllm_exl3_C";
}
