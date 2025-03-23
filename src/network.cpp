module;

#include <torch/torch.h>

export module damathzero:network;

namespace DamathZero {

export using Value = torch::Tensor;
export using Policy = torch::Tensor;

export struct Network : public torch::nn::Module {
  torch::nn::Linear fc1, fc2, value_head, policy_head;

  Network()
      : fc1(register_module("fc1", torch::nn::Linear(9, 16))),
        fc2(register_module("fc2", torch::nn::Linear(16, 32))),
        value_head(register_module("value", torch::nn::Linear(32, 1))),
        policy_head(register_module("policy", torch::nn::Linear(32, 9))) {}

  auto forward(torch::Tensor x) -> std::tuple<Value, Policy> {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));

    auto value = torch::tanh(value_head->forward(x));
    auto policy = policy_head->forward(x);

    return {value, policy};
  }
};

export struct UniformNetwork : public torch::nn::Module {
  auto forward(torch::Tensor x) -> std::tuple<Value, Policy> {
    auto uniform_value = torch::tensor({0.5});
    auto uniform_policy = torch::softmax(torch::ones_like(x), -1);
    return {uniform_value, uniform_policy};
  }
};

}  // namespace DamathZero