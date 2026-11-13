#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
void launch(const void*, const void*, const void*, const void*, const void*, const void*, void*, float*, int, int, int, cudaStream_t);
at::Tensor run(at::Tensor r, at::Tensor w, at::Tensor k, at::Tensor v,
    at::Tensor a, at::Tensor b, at::Tensor state) {
    TORCH_CHECK(r.dim() == 3 && r.size(0) > 0 && r.size(1) > 0 && r.size(2) % 64 == 0,
        "Expected nonempty [batch,time,width] with width divisible by 64");
    for (const auto& x : {r,w,k,v,a,b}) {
        TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16
            && x.sizes() == r.sizes() && x.device() == r.device(), "Invalid recurrence input");
    }
    int B = r.size(0), T = r.size(1), H = r.size(2) / 64;
    TORCH_CHECK(state.is_cuda() && state.device() == r.device() && state.is_contiguous()
        && state.scalar_type() == at::kFloat && state.dim() == 4 && state.size(0) == B
        && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64, "Invalid state");
    c10::cuda::CUDAGuard guard(r.device());
    auto y = at::empty_like(r);
    launch(r.data_ptr(), w.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(), y.data_ptr(), state.data_ptr<float>(), B, T, H, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
TORCH_LIBRARY(basin_recurrent_v1, m) {
    m.def("forward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) state) -> Tensor");
}
TORCH_LIBRARY_IMPL(basin_recurrent_v1, CUDA, m) { m.impl("forward", &run); }
