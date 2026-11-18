// Operator bindings and input checks for master_adamw_kernel.cu.
//
// `step` takes every tensor list on each call. Converting some 4,000 Python
// tensors per update costs about 3 ms, so the trainer instead registers its
// lists once (`create_plan`) and then calls `run_plan` with scalars only; the
// current gradients are read from the model weights here.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace basin {
void master_adamw_launch(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    bool bfloat16_weights,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps);
}

namespace {

struct Plan {
  std::vector<at::Tensor> masters, exp_avgs, exp_avg_sqs, state_steps, weights;
};

std::mutex plans_mutex;
std::unordered_map<int64_t, std::shared_ptr<const Plan>> plans;
int64_t next_plan = 1;

// Checks that do not depend on the gradient: one device, float32 state, one
// model-weight dtype, matching sizes, contiguous memory.
void check_plan(const Plan& plan) {
  const auto count = plan.masters.size();
  TORCH_CHECK(
      plan.exp_avgs.size() == count && plan.exp_avg_sqs.size() == count &&
          plan.state_steps.size() == count && plan.weights.size() == count,
      "master_adamw: tensor lists differ in length");
  if (count == 0) {
    return;
  }
  const auto device = plan.masters[0].device();
  const auto weight_type = plan.weights[0].scalar_type();
  TORCH_CHECK(device.is_cuda(), "master_adamw: tensors must be on CUDA");
  TORCH_CHECK(
      weight_type == at::kBFloat16 || weight_type == at::kFloat,
      "master_adamw: model weights must be bfloat16 or float32, got ",
      weight_type);
  for (size_t i = 0; i < count; i++) {
    const auto& master = plan.masters[i];
    const auto& weight = plan.weights[i];
    for (const at::Tensor* state :
         {&master, &plan.exp_avgs[i], &plan.exp_avg_sqs[i]}) {
      TORCH_CHECK(
          state->device() == device && state->scalar_type() == at::kFloat &&
              state->is_contiguous() && state->numel() == master.numel(),
          "master_adamw: master weights and moments must be contiguous float32 "
          "tensors of equal size on one device (position ",
          i,
          ")");
    }
    TORCH_CHECK(
        weight.device() == device && weight.scalar_type() == weight_type &&
            weight.is_contiguous() && weight.sizes() == master.sizes(),
        "master_adamw: model weight does not match its master (position ",
        i,
        ")");
    TORCH_CHECK(
        plan.state_steps[i].device() == device &&
            plan.state_steps[i].scalar_type() == at::kFloat &&
            plan.state_steps[i].numel() == 1,
        "master_adamw: step counters must be float32 scalars on the device");
  }
}

// Updates every position whose gradient is defined; the others keep their
// weights, moments and step count, as torch.optim.AdamW does. Returns the
// number of tensors updated.
int64_t apply(
    const Plan& plan,
    const std::vector<at::Tensor>& gradients,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps) {
  std::vector<std::vector<at::Tensor>> lists(5);
  std::vector<at::Tensor> steps;
  for (size_t i = 0; i < plan.masters.size(); i++) {
    at::Tensor gradient = gradients[i];
    if (!gradient.defined()) {
      continue;
    }
    const auto& weight = plan.weights[i];
    TORCH_CHECK(
        gradient.device() == weight.device() &&
            gradient.scalar_type() == weight.scalar_type() &&
            gradient.sizes() == weight.sizes(),
        "master_adamw: gradient does not match its model weight (position ",
        i,
        ")");
    if (!gradient.is_contiguous()) {
      // The kernel walks memory linearly; same values, in the weight's order.
      gradient = gradient.contiguous();
    }
    lists[0].push_back(plan.masters[i]);
    lists[1].push_back(std::move(gradient));
    lists[2].push_back(plan.exp_avgs[i]);
    lists[3].push_back(plan.exp_avg_sqs[i]);
    lists[4].push_back(weight);
    steps.push_back(plan.state_steps[i]);
  }
  if (steps.empty()) {
    return 0;
  }
  c10::cuda::CUDAGuard guard(plan.masters[0].device());
  // As torch.optim.AdamW: advance the step counters, then update.
  at::_foreach_add_(steps, 1);
  basin::master_adamw_launch(
      lists,
      steps,
      plan.weights[0].scalar_type() == at::kBFloat16,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps);
  // These operators have no autograd kernel, so record the in-place writes
  // here: a graph that saved an older value must fail, not silently use the
  // new one.
  for (const int list : {0, 2, 3, 4}) {
    for (const auto& tensor : lists[list]) {
      tensor.unsafeGetTensorImpl()->bump_version();
    }
  }
  return static_cast<int64_t>(steps.size());
}

Plan make_plan(
    at::TensorList masters,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    at::TensorList weights) {
  Plan plan{
      masters.vec(),
      exp_avgs.vec(),
      exp_avg_sqs.vec(),
      state_steps.vec(),
      weights.vec()};
  check_plan(plan);
  return plan;
}

int64_t master_adamw_step(
    at::TensorList masters,
    const c10::List<std::optional<at::Tensor>>& gradients,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    at::TensorList weights,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps) {
  const Plan plan = make_plan(masters, exp_avgs, exp_avg_sqs, state_steps, weights);
  TORCH_CHECK(
      gradients.size() == plan.masters.size(),
      "master_adamw: tensor lists differ in length");
  std::vector<at::Tensor> values;
  values.reserve(gradients.size());
  for (size_t i = 0; i < gradients.size(); i++) {
    const std::optional<at::Tensor> gradient = gradients.get(i);
    values.push_back(gradient.has_value() ? *gradient : at::Tensor());
  }
  return apply(plan, values, lr, beta1, beta2, weight_decay, eps);
}

int64_t master_adamw_create_plan(
    at::TensorList masters,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    at::TensorList weights) {
  auto plan = std::make_shared<const Plan>(
      make_plan(masters, exp_avgs, exp_avg_sqs, state_steps, weights));
  std::lock_guard<std::mutex> lock(plans_mutex);
  const int64_t id = next_plan++;
  plans.emplace(id, std::move(plan));
  return id;
}

int64_t master_adamw_run_plan(
    int64_t id,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps) {
  std::shared_ptr<const Plan> plan;
  {
    std::lock_guard<std::mutex> lock(plans_mutex);
    const auto found = plans.find(id);
    TORCH_CHECK(found != plans.end(), "master_adamw: unknown plan ", id);
    plan = found->second;
  }
  std::vector<at::Tensor> gradients;
  gradients.reserve(plan->weights.size());
  for (const auto& weight : plan->weights) {
    gradients.push_back(weight.grad());
  }
  return apply(*plan, gradients, lr, beta1, beta2, weight_decay, eps);
}

void master_adamw_release_plan(int64_t id) {
  std::lock_guard<std::mutex> lock(plans_mutex);
  plans.erase(id);
}

}  // namespace

TORCH_LIBRARY(basin_master_adamw_v1, m) {
  m.def(
      "step(Tensor(a!)[] masters, Tensor?[] gradients, Tensor(b!)[] exp_avgs, "
      "Tensor(c!)[] exp_avg_sqs, Tensor(d!)[] state_steps, Tensor(e!)[] weights, "
      "float lr, float beta1, float beta2, float weight_decay, float eps) -> int",
      &master_adamw_step);
  m.def(
      "create_plan(Tensor[] masters, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, "
      "Tensor[] state_steps, Tensor[] weights) -> int",
      &master_adamw_create_plan);
  m.def(
      "run_plan(int plan, float lr, float beta1, float beta2, "
      "float weight_decay, float eps) -> int",
      &master_adamw_run_plan);
  m.def("release_plan(int plan) -> ()", &master_adamw_release_plan);
}
