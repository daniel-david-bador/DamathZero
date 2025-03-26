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
  MCTS(Config config) : config_(config) {}

  template <typename Network>
  constexpr auto search(Board board, std::shared_ptr<Network> network)
      -> torch::Tensor {
    torch::NoGradGuard no_grad;

    auto root_id = nodes_.create(board);

    for (auto _ : std::views::iota(0, config_.NumSimulations)) {
      auto node = nodes_.as_ref(root_id);

      while (not is_leaf(node.id))
        node = highest_child_score(node.id);

      auto [value, terminal] =
          Game::get_value_and_terminated(node->board, node->action);
      value = Game::get_opponent_value(value);

      if (not terminal) {
        auto [value_tensor, policy] = network->forward(
            torch::unsqueeze(torch::tensor(node->board, torch::kFloat32), 0));
        policy = torch::softmax(torch::squeeze(policy, 0), -1);
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
      child_visits[child.action] = auto(child.visits);
    }

    nodes_.clear();

    return child_visits / child_visits.sum(0);
  }

  constexpr auto is_leaf(Node::ID id) const -> bool {
    auto& node = nodes_.get(id);
    return node.children.empty();
  };

  constexpr auto score(Node::ID id) const -> double {
    auto& child = nodes_.get(id);
    auto& parent = nodes_.get(child.parent);

    auto mean = child.visits > 0.0
                    ? 1.0 - ((child.value / child.visits) + 1) / 2.0
                    : 0.0;
    return mean + child.prior * config_.C *
                      (std::sqrt(parent.visits) / (1 + child.visits));
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
    auto contiguous_policy = policy.contiguous();
    auto priors = std::vector<double>(
        contiguous_policy.data_ptr<float>(),
        contiguous_policy.data_ptr<float>() + contiguous_policy.numel());
    auto legal_actions = Game::legal_actions(parent->board);
    assert(priors.size() == legal_actions.size());
    for (auto [prior, action] : std::views::zip(priors, legal_actions)) {
      auto [child_board, _] = Game::apply_action(parent->board, action, 1);
      child_board = Game::change_perspective(child_board, -1);
      parent.create_child(child_board, action, prior);
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
  Config config_;
};

}  // namespace DamathZero
