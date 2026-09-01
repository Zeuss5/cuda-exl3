#include <torch/extension.h>
#include <torch/library.h>

namespace vllm_exl3 {
void exl3_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh);
void exl3_reserve(const at::Tensor& like, int64_t max_tokens, int64_t k, int64_t n,
                  int64_t groups);
void exl3_moe_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh,
                     const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                     const at::Tensor& n_rows, int64_t block_m, int64_t top_k,
                     int64_t m_valid);
at::Tensor exl3_moe_gemm(const at::Tensor& a_had, const at::Tensor& trellis,
                         const at::Tensor& suh_unused, const at::Tensor& svh,
                         const at::Tensor& expert_ids, const at::Tensor& n_rows,
                         at::IntArrayRef group_n, int64_t cb, int64_t block_m,
                         at::ScalarType out_dtype);
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
    m.def(
        "exl3_moe_had_in(Tensor x, Tensor(a!) out, Tensor suh, Tensor sorted_ids, "
        "Tensor expert_ids, Tensor n_rows, int block_m, int top_k, int m_valid) -> ()");
    m.def(
        "exl3_moe_gemm(Tensor a_had, Tensor trellis, Tensor suh, Tensor svh, "
        "Tensor expert_ids, Tensor n_rows, int[] group_n, int cb, int block_m, "
        "ScalarType out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(vllm_exl3_C, CUDA, m)
{
    m.impl("exl3_linear", &vllm_exl3::exl3_linear);
    m.impl("exl3_had_in", &vllm_exl3::exl3_had_in);
    m.impl("exl3_reserve", &vllm_exl3::exl3_reserve);
    m.impl("exl3_moe_had_in", &vllm_exl3::exl3_moe_had_in);
    m.impl("exl3_moe_gemm", &vllm_exl3::exl3_moe_gemm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "EXL3 kernels; ops are registered under torch.ops.vllm_exl3_C";
}
