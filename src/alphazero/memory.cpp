#include "alphazero/memory.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <span>

namespace az {

auto Memory::size() -> size_t { return data_.size(); }

auto Memory::pop() -> void { data_.pop_back(); }

auto Memory::shuffle() -> void { std::ranges::shuffle(data_, gen_); }

auto Memory::append(Feature feature, Value value, Policy policy) -> void {
  data_.emplace_back(feature, value, policy);
}

auto Memory::sample_batch(std::size_t batch_size, std::size_t start)
    -> std::tuple<Feature, Value, Policy> {
  auto size = std::min(batch_size, data_.size() - start);

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

  return {torch::stack(features, 0), torch::stack(values, 0),
          torch::stack(policies, 0)};
}

}  // namespace az
