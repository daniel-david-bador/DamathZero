module;

#include <assert.h>
#include <torch/torch.h>

export module damathzero:mcts;

import std;

import :config;
import :node;
import :storage;
import :game;

namespace DamathZero {

export template <Concepts::Game Game>
class MCTS {
 public:
  MCTS(Config config) : config_(config) {}

  constexpr auto search(Game::State original_state,
                        std::shared_ptr<typename Game::Network> model)
      -> torch::Tensor {
    torch::NoGradGuard no_grad;

    auto root_id = nodes_.create(original_state.player);

    for (auto _ : std::views::iota(0, config_.NumSimulations)) {
      auto node = nodes_.as_ref(root_id);
      auto state = original_state;

      while (not is_leaf(node.id)) {
        node = highest_child_score(node.id);
        state = Game::apply_action(state, node->action);
      }

      if (auto terminal_value = Game::terminal_value(state, node->action);
          terminal_value.has_value()) {
        auto& parent = nodes_.get(node->parent_id);
        backpropagate(node.id, *terminal_value, parent.player);
      } else {
        auto value = expand(node.id, state, model);
        backpropagate(node.id, value, state.player);
      }
    }

    auto child_visits = torch::zeros(Game::ActionSize, torch::kFloat32);
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
    auto& parent = nodes_.get(child.parent_id);

    assert(child.player != parent.player);

    auto exploration = child.prior * config_.C *
                       (std::sqrt(parent.visits) / (1 + child.visits));

    if (child.visits == 0)
      return exploration;

    auto mean = ((child.value / child.visits) + 1) / 2.0;
    if (child.player != parent.player)
      mean = 1 - mean;

    return mean + exploration;
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

  constexpr auto expand(Node::ID parent_id, const Game::State& state,
                        std::shared_ptr<typename Game::Network> model)
      -> double {
    auto feature = Game::encode_state(state);
    auto legal_actions = Game::legal_actions(state);

    auto [value, policy] = model->forward(torch::unsqueeze(feature, 0));
    policy = torch::softmax(torch::squeeze(policy, 0), -1);
    policy *= legal_actions;
    policy /= policy.sum();

    auto parent = nodes_.as_ref(parent_id);
    assert(parent->player == state.player);

    auto actions = legal_actions.nonzero();
    for (auto i = 0; i < actions.size(0); i++) {
      auto action = actions[i].template item<Action>();
      auto prior = policy[action].template item<double>();
      auto new_state = Game::apply_action(state, action);
      assert(new_state.player != state.player);
      parent.create_child(new_state.player, action, prior);
    }

    return value.template item<double>();
  };

  constexpr auto backpropagate(Node::ID node_id, double value, Player player)
      -> void {
    while (node_id != -1) {
      auto& node = nodes_.get(node_id);

      node.visits += 1;

      if (node.player == player) {
        node.value += value;
      } else {
        node.value -= value;
      }

      node_id = node.parent_id;
    };
  };

 private:
  NodeStorage nodes_;
  Config config_;
};

}  // namespace DamathZero
