module;

#include <assert.h>
#include <torch/torch.h>

export module az:mcts;

import :model;
import :node;
import :storage;
import :game;

import std;

namespace az {

export template <concepts::Game Game, concepts::Model Model>
class MCTS {
 public:
  struct Config {
    int32_t num_simulations = 100;

    float32_t C = 2.0;

    float32_t dirichlet_alpha = 0.3;
    float32_t dirichlet_epsilon = 0.25;

    float32_t temperature = 1.25;
  };

  MCTS(Config config) : config_(config) {}

  constexpr auto search(Game::State original_state,
                        std::shared_ptr<Model> model,
                        std::optional<int> num_simulations = std::nullopt,
                        std::optional<std::mt19937*> noise_gen = std::nullopt)
      -> torch::Tensor {
    torch::NoGradGuard no_grad;

    auto root_id = nodes_.create(original_state.player);
    if (noise_gen) {
      expand(root_id, original_state, model);
      add_exploration_noise(root_id, *noise_gen);
    }

    num_simulations = num_simulations.value_or(config_.num_simulations);
    for (auto _ : std::views::iota(0, *num_simulations + 1)) {
      auto node = nodes_.as_ref(root_id);
      auto state = original_state;

      while (node->is_expanded()) {
        node = highest_child_score(node.id);
        state = Game::apply_action(state, node->action);
      }

      if (auto outcome = Game::get_outcome(state, node->action)) {
        auto& parent = nodes_.get(node->parent_id);
        backpropagate(node.id, outcome->as_scalar(), parent.player);
      } else {
        auto value = expand(node.id, state, model);
        backpropagate(node.id, value, state.player);
      }
    }

    auto child_visits = torch::zeros(Game::ActionSize, torch::kFloat32);
    for (auto child_id : nodes_.get(root_id).children()) {
      auto& child = nodes_.get(child_id);
      child_visits[child.action] = auto(child.visits);
    }

    nodes_.clear();

    return child_visits / child_visits.sum(0);
  }

 private:
  constexpr auto score(NodeId id) const -> double {
    auto& child = nodes_.get(id);
    auto& parent = nodes_.get(child.parent_id);

    auto exploration = child.prior * config_.C *
                       (std::sqrt(parent.visits) / (1 + child.visits));

    if (child.visits == 0)
      return exploration;

    auto mean = ((child.value / child.visits) + 1) / 2.0;
    if (child.player != parent.player)
      mean = 1 - mean;

    return mean + exploration;
  };

  constexpr auto highest_child_score(NodeId id) const -> NodeId {
    auto& node = nodes_.get(id);
    assert(node.is_expanded());

    auto highest_score = [this](NodeId a, NodeId b) {
      return score(a) < score(b);
    };

    auto range = node.children();
    return *std::ranges::max_element(range, highest_score);
  };

  constexpr auto highest_child_visits(NodeId id) const -> NodeId {
    auto& node = nodes_.get(id);
    assert(node.is_expanded());

    auto highest_visits = [this](NodeId a, NodeId b) {
      return nodes_.get(a).visits < nodes_.get(b).visits;
    };

    auto range = node.children();
    return *std::ranges::max_element(range, highest_visits);
  };

  constexpr auto expand(NodeId parent_id, const Game::State& state,
                        std::shared_ptr<Model> model) -> double {
    torch::NoGradGuard no_grad;

    auto feature = Game::encode_state(state);
    auto legal_actions = Game::legal_actions(state);

    auto [wdl, policy] = model->forward(torch::unsqueeze(feature, 0));
    policy = torch::softmax(torch::squeeze(policy, 0), -1);
    policy *= legal_actions;
    policy /= policy.sum();

    auto parent = nodes_.as_ref(parent_id);

    auto actions = legal_actions.nonzero();
    for (auto i = 0; i < actions.size(0); i++) {
      auto action = actions[i].template item<Action>();
      auto prior = policy[action].template item<double>();
      auto new_state = Game::apply_action(state, action);
      parent.create_child(new_state.player, action, prior);
    }

    wdl = wdl.squeeze(0);

    auto value = wdl[0] - wdl[2];
    return value.template item<double>();
  };

  constexpr auto backpropagate(NodeId node_id, double value, Player player)
      -> void {
    while (node_id.is_valid()) {
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

  constexpr auto add_exploration_noise(NodeId node_id, std::mt19937* gen)
      -> void {
    assert(gen != nullptr);

    auto& node = nodes_.get(node_id);

    auto epsilon = config_.dirichlet_epsilon;
    auto noise = std::vector<float32_t>(node.num_children(), 0.0);
    auto gamma =
        std::gamma_distribution<float32_t>(config_.dirichlet_alpha, 1.0);

    auto sum = 0.0;
    std::ranges::generate(noise, [&sum, &gamma, gen] {
      auto x = gamma(*gen);
      sum += x;
      return x;
    });

    for (auto [child_id, x] : std::views::zip(node.children(), noise)) {
      auto& child = nodes_.get(child_id);
      x = x / sum;
      child.prior = child.prior * (1 - epsilon) + x * epsilon;
    }
  }

 private:
  NodeStorage nodes_;
  Config config_;
};

}  // namespace az
