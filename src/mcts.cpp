module;

#include <assert.h>
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
  template <typename Network>
  constexpr auto search(Board board, std::shared_ptr<Network> network)
      -> torch::Tensor {
    torch::NoGradGuard no_grad;

    auto root_id = nodes_.create(board, Player{1});

    for (auto _ : std::views::iota(0, Config::NumSimulations)) {
      auto node = nodes_.as_ref(root_id);

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

    auto child_visits = torch::zeros(9, torch::kFloat32);
    for (auto child_id : nodes_.get(root_id).children) {
      auto& child = nodes_.get(child_id);
      child_visits[child.action] = child.visits;
    }

    nodes_.clear();

    return child_visits / child_visits.sum(0);
  }

  constexpr auto is_leaf(Node::ID id) const -> bool {
    auto& node = nodes_.get(id);
    return node.children.empty();
  };

  constexpr auto score(Node::ID id) const -> double {
    auto& node = nodes_.get(id);
    auto& parent = nodes_.get(node.parent);

    auto mean =
        node.visits > 0.0 ? 1 - ((node.value / node.visits) + 1) / 2.0 : 0.0;
    return mean + node.prior * 2 * std::sqrt(parent.visits) / (1 + node.visits);
  };

  constexpr auto highest_child_score(Node::ID id) const -> Node::ID {
    auto& node = nodes_.get(id);
    assert(not node.children.empty());

    auto highest_score = [this](Node::ID a, Node::ID b) {
      return score(a) < score(b);
    };

    return *std::ranges::max_element(node.children, highest_score);
  };

  constexpr auto highest_child_visits(Node::ID id) const -> Node::ID {
    auto& node = nodes_.get(id);
    assert(not node.children.empty());

    auto highest_visits = [this](Node::ID a, Node::ID b) {
      return nodes_.get(a).visits < nodes_.get(b).visits;
    };

    return *std::ranges::max_element(node.children, highest_visits);
  };

  constexpr auto expand(Node::ID parent_id, const Policy& policy) -> void {
    auto parent = nodes_.as_ref(parent_id);
    auto legal_actions = Game::legal_actions(parent->board);
    for (size_t i = 0; i < legal_actions.size(); i++) {
      auto action = legal_actions[i];
      auto prior = policy[i].item<double>();
      auto [child_board, player] = Game::apply_action(parent->board, action, parent->player);
      child_board = Game::change_perspective(child_board, player);
      parent.create_child(child_board, player, action, prior);
    }
  };

  constexpr auto backpropagate(Node::ID id, double value) -> void {
    auto& node = nodes_.get(id);

    node.visits += 1;
    node.value += value;

    value = Game::get_opponent_value(value);
    if (node.parent > 0)
      backpropagate(node.parent, value);
  };

  NodeStorage nodes_;
};

}  // namespace DamathZero
