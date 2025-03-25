module;

#include <torch/torch.h>

export module damathzero;

import std;

export import :game;
export import :memory;
export import :mcts;
export import :network;
export import :node;
export import :storage;

namespace DamathZero {

export template <typename Network>
class DamathZero {
 public:
  DamathZero(std::shared_ptr<Network> model,
             std::shared_ptr<torch::optim::Optimizer> optimizer,
             std::random_device& rd)
      : model_(model), optimizer_(optimizer), rd_(rd) {}

  auto learn() -> void {
    auto memory = Memory{rd_};

    for (auto _ : std::views::iota(0, Config::NumIterations)) {
      model_->eval();
      for (auto _ : std::views::iota(0, Config::NumSelfPlayIterations))
        memory.merge(generate_self_play_data());

      model_->train();
      for (auto _ : std::views::iota(0, Config::NumTrainingEpochs))
        train(memory);

      // torch::save(model_, std::format("models/model_{}.pt", i));
      // torch::save(optimizer_, std::format("optimizer_{}.pt", i));
    }
  }

  auto train(Memory& memory) {
    namespace F = torch::nn::functional;
    for (size_t i = 0; i < memory.size(); i += Config::BatchSize) {
      auto [feature, target_value, target_policy] = memory.sample_batch(i);
      auto [out_value, out_policy] = model_->forward(feature);

      auto loss = F::mse_loss(out_value, target_value) +
                  F::cross_entropy(out_policy, target_policy);

      optimizer_->zero_grad();
      loss.backward();
      optimizer_->step();
    }
  }

  auto generate_self_play_data() -> Memory {
    auto nodes = std::make_shared<NodeStorage>();
    auto mcts = MCTS{nodes};

    auto node = nodes->as_ref(nodes->create());

    auto [value, terminal] =
        Game::get_value_and_terminated(node->board, node->action);

    auto path = std::vector<Node::ID>{};

    while (not terminal) {
      path.push_back(node.id);
      node = mcts.search(node.id, model_);
      nodes->detach(node.id);

      std::tie(value, terminal) =
          Game::get_value_and_terminated(node->board, node->action);
    }
    path.pop_back();

    auto memory = Memory{rd_};

    for (auto node_id : path) {
      auto current_node = nodes->as_ref(node_id);
      auto target_feature = torch::tensor(current_node->board, torch::kFloat32);
      auto target_value =
          torch::tensor({current_node->player == node->player ? value : -value},
                        torch::kFloat32);
      auto target_policy = mcts.get_action_probs(node_id);
      memory.append(target_feature, target_value, target_policy);
    }

    return memory;
  }

 private:
  std::shared_ptr<Network> model_;
  std::shared_ptr<torch::optim::Optimizer> optimizer_;
  std::random_device& rd_;
};

}  // namespace DamathZero
