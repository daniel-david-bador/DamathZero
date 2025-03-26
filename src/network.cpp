module;

#include <torch/torch.h>

export module damathzero:network;

namespace DamathZero {

export using Feature = torch::Tensor;
export using Value = torch::Tensor;
export using Policy = torch::Tensor;

export struct DynamicTanH : public torch::nn::Module {
  auto forward(torch::Tensor x) -> torch::Tensor { return torch::tanh(x); }
};

export struct Network : public torch::nn::Module {
  torch::nn::Linear fc1, fc2, value_head, policy_head;
  torch::nn::BatchNorm1d bn1, bn2;

  Network()
      : fc1(register_module("fc1", torch::nn::Linear(9, 64))),
        fc2(register_module("fc2", torch::nn::Linear(64, 32))),
        value_head(register_module("value", torch::nn::Linear(32, 1))),
        policy_head(register_module("policy", torch::nn::Linear(32, 9))),
        bn1(register_module("bn1", torch::nn::BatchNorm1d(64))),
        bn2(register_module("bn2", torch::nn::BatchNorm1d(32))) {}

  auto forward(Feature x) -> std::tuple<Value, Policy> {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(bn1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(bn2->forward(x));

    auto value = torch::tanh(value_head->forward(x));
    auto policy = policy_head->forward(x);

    return {value, policy};
  }
};

export struct UniformNetwork : public torch::nn::Module {
  auto forward(Feature x) -> std::tuple<Value, Policy> {
    auto uniform_value = torch::tensor({0.5});
    auto uniform_policy = torch::softmax(torch::ones_like(x), -1);
    return {uniform_value, uniform_policy};
  }
};

}  // namespace DamathZero
