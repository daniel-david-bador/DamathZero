module;

#include <torch/torch.h>

export module alphazero:memory;

import std;

import :config;
import :game;

namespace AZ {

export using Feature = torch::Tensor;
export using Policy = torch::Tensor;
export using Value = torch::Tensor;

export class Memory {
 public:
  Memory(Config config, std::mt19937& gen) : config_{config}, gen_(gen) {}

  constexpr auto size() -> size_t { return data_.size(); }

  constexpr auto pop() -> void {
    auto guard = std::lock_guard(mutex_);
    data_.pop_back();
  }

  constexpr auto shuffle() -> void {
    auto guard = std::lock_guard(mutex_);
    std::ranges::shuffle(data_, gen_);
  }

  auto append(Feature feature, Value value, Policy policy) -> void {
    auto guard = std::lock_guard(mutex_);

    data_.emplace_back(feature, value, policy);
  }

  auto sample_batch(std::size_t start) -> std::tuple<Feature, Value, Policy> {
    auto guard = std::lock_guard(mutex_);

    auto size = std::min(config_.batch_size, data_.size() - start);

    auto batch = std::span{data_.begin() + start, data_.begin() + start + size};

    // // TODO: investigate why this invariant is invalidated sometimes which
    // causes the batch norm to throw an exception.
    // assert(size > 1);

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

    auto device = config_.device;

    return {torch::stack(features, 0).to(device),
            torch::stack(values, 0).to(device),
            torch::stack(policies, 0).to(device)};
  }

 private:
  std::mutex mutex_;
  Config config_;
  std::mt19937& gen_;
  std::vector<std::tuple<Feature, Value, Policy>> data_;
};

}  // namespace AZ
