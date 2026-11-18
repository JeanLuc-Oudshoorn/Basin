// One-pass AdamW for FP32 master weights with low-precision model weights.
//
// The Python fallback (master_optimizer.py) makes three passes over every
// parameter: cast the gradient to float32, run PyTorch's fused AdamW on the
// float32 master, copy the master back into the model weight. This kernel
// does the same arithmetic in one pass: it reads the gradient in the model's
// dtype, updates master and moments in float32, and writes the rounded model
// weight.
//
// The update itself is PyTorch's own `adam_math` template, instantiated for
// float32 exactly as `torch._fused_adamw_` instantiates it, and the bias
// corrections are computed as FusedAdamMathFunctor computes them. Only the
// loads and stores differ, and both conversions are exact or identical to
// `Tensor.copy_` (bfloat16 -> float32 is exact; float32 -> bfloat16 uses the
// same c10 round-to-nearest-even cast). Do not compile with --use_fast_math:
// results are meant to match the fallback bit for bit.
#include <ATen/ATen.h>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <c10/util/BFloat16.h>

namespace basin {

using at::native::ADAM_MODE;
using at::native::FusedOptimizerTensorListMetadata;

constexpr int kDepth = 5;  // master, gradient, exp_avg, exp_avg_sq, model weight
constexpr int kWeightIdx = 4;

template <typename weight_t>
struct MasterAdamWFunctor {
  using opmath_t = float;

  C10_DEVICE __forceinline__ void operator()(
      int64_t chunk_size,
      FusedOptimizerTensorListMetadata<kDepth>& tl,
      const double& lr,
      const double& beta1,
      const double& beta2,
      const double& weight_decay,
      const double& eps) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

    const auto lr_opmath = static_cast<opmath_t>(lr);
    const auto beta1_opmath = static_cast<opmath_t>(beta1);
    const auto beta2_opmath = static_cast<opmath_t>(beta2);
    const auto weight_decay_opmath = static_cast<opmath_t>(weight_decay);
    const auto eps_opmath = static_cast<opmath_t>(eps);

    const auto step_count = static_cast<opmath_t>(*reinterpret_cast<const float*>(
        tl.state_steps_addresses[tensor_loc]));
    const opmath_t bias_correction1 =
        1 - at::native::pow_(beta1_opmath, step_count);
    const opmath_t bias_correction2 =
        1 - at::native::pow_(beta2_opmath, step_count);
    const opmath_t bias_correction2_sqrt = std::sqrt(bias_correction2);

    const auto offset = chunk_idx * chunk_size;
    const auto n = tl.numel_for_tensor[tensor_loc] - offset;
    auto* master =
        (float*)tl.addresses[at::native::kParamIdx][tensor_loc] + offset;
    const auto* gradient =
        (const weight_t*)tl.addresses[at::native::kGradIdx][tensor_loc] + offset;
    auto* exp_avg =
        (float*)tl.addresses[at::native::kExpAvgIdx][tensor_loc] + offset;
    auto* exp_avg_sq =
        (float*)tl.addresses[at::native::kExpAvgSqIdx][tensor_loc] + offset;
    auto* weight = (weight_t*)tl.addresses[kWeightIdx][tensor_loc] + offset;

    float r_args[4][at::native::kILP];
    const bool vector_access =
        n % at::native::kILP == 0 && chunk_size % at::native::kILP == 0 &&
        at::native::is_aligned(master) && at::native::is_aligned(gradient) &&
        at::native::is_aligned(exp_avg) && at::native::is_aligned(exp_avg_sq) &&
        at::native::is_aligned(weight);
    if (vector_access) {
      // Four neighbouring elements per thread, moved as one aligned vector, as
      // PyTorch's fused optimizers do. The arithmetic per element is the same
      // as in the general loop below.
      for (int64_t i_start = threadIdx.x;
           i_start * at::native::kILP < n &&
           i_start * at::native::kILP < chunk_size;
           i_start += blockDim.x) {
        weight_t narrow[at::native::kILP];
        at::native::load_store(r_args[at::native::kParamIdx], master, 0, i_start);
        at::native::load_store(
            narrow, const_cast<weight_t*>(gradient), 0, i_start);
        at::native::load_store(r_args[at::native::kExpAvgIdx], exp_avg, 0, i_start);
        at::native::load_store(
            r_args[at::native::kExpAvgSqIdx], exp_avg_sq, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < at::native::kILP; ii++) {
          r_args[at::native::kGradIdx][ii] = static_cast<float>(narrow[ii]);
        }
        at::native::adam_math<float, opmath_t, 4, ADAM_MODE::ADAMW, false>(
            r_args,
            lr_opmath,
            beta1_opmath,
            beta2_opmath,
            weight_decay_opmath,
            eps_opmath,
            false,    // maximize
            nullptr,  // grad_scale
            nullptr,  // found_inf
            bias_correction1,
            bias_correction2_sqrt);
#pragma unroll
        for (int ii = 0; ii < at::native::kILP; ii++) {
          narrow[ii] = static_cast<weight_t>(r_args[at::native::kParamIdx][ii]);
        }
        at::native::load_store(master, r_args[at::native::kParamIdx], i_start, 0);
        at::native::load_store(exp_avg, r_args[at::native::kExpAvgIdx], i_start, 0);
        at::native::load_store(
            exp_avg_sq, r_args[at::native::kExpAvgSqIdx], i_start, 0);
        at::native::load_store(weight, narrow, i_start, 0);
      }
      return;
    }
    for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
         i_start += blockDim.x * at::native::kILP) {
#pragma unroll
      for (int ii = 0; ii < at::native::kILP; ii++) {
        const auto i = i_start + threadIdx.x + ii * blockDim.x;
        const bool live = i < n && i < chunk_size;
        r_args[at::native::kParamIdx][ii] = live ? master[i] : 0;
        r_args[at::native::kGradIdx][ii] =
            live ? static_cast<float>(gradient[i]) : 0;
        r_args[at::native::kExpAvgIdx][ii] = live ? exp_avg[i] : 0;
        r_args[at::native::kExpAvgSqIdx][ii] = live ? exp_avg_sq[i] : 0;
      }
      at::native::adam_math<float, opmath_t, 4, ADAM_MODE::ADAMW, false>(
          r_args,
          lr_opmath,
          beta1_opmath,
          beta2_opmath,
          weight_decay_opmath,
          eps_opmath,
          false,    // maximize
          nullptr,  // grad_scale
          nullptr,  // found_inf
          bias_correction1,
          bias_correction2_sqrt);
#pragma unroll
      for (int ii = 0; ii < at::native::kILP; ii++) {
        const auto i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          master[i] = r_args[at::native::kParamIdx][ii];
          exp_avg[i] = r_args[at::native::kExpAvgIdx][ii];
          exp_avg_sq[i] = r_args[at::native::kExpAvgSqIdx][ii];
          weight[i] = static_cast<weight_t>(r_args[at::native::kParamIdx][ii]);
        }
      }
    }
  }
};

// tensor_lists: {masters, gradients, exp_avgs, exp_avg_sqs, weights}, all on
// one CUDA device, contiguous, and of equal element count per position.
void master_adamw_launch(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    bool bfloat16_weights,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps) {
  if (bfloat16_weights) {
    at::native::multi_tensor_apply_for_fused_optimizer<kDepth>(
        tensor_lists,
        state_steps,
        MasterAdamWFunctor<c10::BFloat16>(),
        lr,
        beta1,
        beta2,
        weight_decay,
        eps);
  } else {
    at::native::multi_tensor_apply_for_fused_optimizer<kDepth>(
        tensor_lists,
        state_steps,
        MasterAdamWFunctor<float>(),
        lr,
        beta1,
        beta2,
        weight_decay,
        eps);
  }
}

}  // namespace basin
