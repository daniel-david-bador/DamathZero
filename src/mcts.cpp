module;

#include <torch/torch.h>

export module damathzero:mcts;

import std;

import :game;
import :network;
import :config;
import :node;
import :storage;

namespace DamathZero {

export class MCTS {
 public:
  MCTS(std::shared_ptr<NodeStorage> nodes) : nodes_(nodes) {};

  template <typename Network>
  constexpr auto search(Node::ID root_id, std::shared_ptr<Network> network)
      -> Node::ID {
    torch::NoGradGuard no_grad;
    for (auto _ : std::views::iota(0, Config::num_simulations)) {
      auto node = nodes_->get_ref(root_id);

      while (not is_leaf(node.id))
        node = highest_child_score(node.id);

      auto [value, terminal] =
          Game::get_value_and_terminated(node->board, node->action);
      value = Game::get_opponent_value(value);

      if (not terminal) {
        auto [value_tensor, policy] =
            network->forward(torch::tensor(node->board, torch::kFloat32));
        policy = torch::softmax(policy, -1);
        policy =
            policy.index({torch::tensor(Game::legal_actions(node->board))});
        policy /= policy.sum();

        value = value_tensor.template item<double>();
        expand(node.id, policy);
      }

      backpropagate(node.id, value);
    }

    return highest_child_visits(root_id);
  }

 private:
  constexpr auto is_leaf(Node::ID id) const -> bool {
    auto& node = nodes_->get(id);
    return node.children.empty();
  };

  constexpr auto score(Node::ID id) const -> double {
    auto& node = nodes_->get(id);
    auto& parent = nodes_->get(node.parent);

    auto mean =
        node.visits > 0.0 ? 1 - ((node.value / node.visits) + 1) / 2.0 : 0.0;
    return mean + node.prior * 2 * std::sqrt(parent.visits) / (1 + node.visits);
  };

  constexpr auto highest_child_score(Node::ID id) -> Node::ID {
    auto& node = nodes_->get(id);
    assert(not node.children.empty());

    auto highest_score = [this](Node::ID a, Node::ID b) {
      return score(a) < score(b);
    };

    return *std::ranges::max_element(node.children, highest_score);
  };

  constexpr auto highest_child_visits(Node::ID id) -> Node::ID {
    auto& node = nodes_->get(id);
    assert(not node.children.empty());

    auto highest_visits = [this](Node::ID a, Node::ID b) {
      return nodes_->get(a).visits < nodes_->get(b).visits;
    };

    return *std::ranges::max_element(node.children, highest_visits);
  };

  constexpr auto expand(Node::ID id, const Policy& policy) -> void {
    auto node = nodes_->get_ref(id);
    auto legal_actions = Game::legal_actions(node->board);
    for (size_t i = 0; i < legal_actions.size(); i++) {
      auto action = legal_actions[i];
      auto prior = policy[i].item<double>();
      auto child_board = Game::apply_action(node->board, action, 1);
      child_board = Game::change_perpective(child_board, -1);
      nodes_->create_child(id, child_board, action, prior, id);
    }
  };

  constexpr auto backpropagate(Node::ID id, double value) -> void {
    auto& node = nodes_->get(id);

    node.visits += 1;
    node.value += value;

    value = Game::get_opponent_value(value);
    if (node.parent > 0)
      backpropagate(node.parent, value);
  };

  std::shared_ptr<NodeStorage> nodes_;
};

}  // namespace DamathZero