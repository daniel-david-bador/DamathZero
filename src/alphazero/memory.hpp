#pragma once

#include <torch/torch.h>

#include <random>

namespace az {

using Feature = torch::Tensor;
using Policy = torch::Tensor;
using Value = torch::Tensor;

class Memory {
 public:
  Memory(std::mt19937& gen) : gen_(gen) {};

  auto size() -> size_t;
  auto pop() -> void;

  auto shuffle() -> void;

  auto append(Feature feature, Value value, Policy policy) -> void;

  auto sample_batch(std::size_t batch_size, std::size_t start)
      -> std::tuple<Feature, Value, Policy>;

 private:
  std::mt19937& gen_;
  std::vector<std::tuple<Feature, Value, Policy>> data_;
};

}  // namespace az
