module;

#include <torch/torch.h>

export module damathzero:memory;

import std;

import :config;
import :game;

namespace DamathZero {

export class Memory {
 public:
  Memory(Config config, std::random_device& device)
      : config_{config}, gen_(device()) {}

  constexpr auto size() -> size_t { return data_.size(); }

  constexpr auto shuffle() -> void { std::ranges::shuffle(data_, gen_); }

  auto append(Feature feature, Value value, Policy policy) -> void {
    data_.emplace_back(feature, value, policy);
  }

  auto merge(Memory&& memory) -> void {
    data_.insert(data_.end(), memory.data_.begin(), memory.data_.end());
  }

  auto sample_batch(std::size_t start) -> std::tuple<Feature, Value, Policy> {
    auto size = std::min(config_.BatchSize, data_.size() - start);

    auto batch = std::span{data_.begin() + start, data_.begin() + start + size};

    std::vector<Feature> features;
    std::vector<Value> values;
    std::vector<Policy> policies;

    features.reserve(size);
    values.reserve(size);
    policies.reserve(size);

    for (auto [feature, value, policy] : batch) {
      features.emplace_back(feature);
      values.emplace_back(value);
      policies.emplace_back(policy);
    }

    return {torch::stack(features, 0), torch::stack(values, 0),
            torch::stack(policies, 0)};
  }

 private:
  Config config_;
  std::mt19937 gen_;
  std::vector<std::tuple<Feature, Value, Policy>> data_;
};

}  // namespace DamathZero
