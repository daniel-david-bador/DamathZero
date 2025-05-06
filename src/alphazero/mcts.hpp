#pragma once

#include <assert.h>
#include <torch/torch.h>

#include <random>
#include <ranges>

#include "alphazero/game.hpp"
#include "alphazero/model.hpp"
#include "alphazero/node.hpp"
#include "alphazero/storage.hpp"

namespace az {

template <concepts::Game Game, concepts::Model Model>
class MCTS {
 public:
  struct Config {
    float32_t C = 2.0;

    float32_t dirichlet_alpha = 0.3;
    float32_t dirichlet_epsilon = 0.25;

    float32_t temperature = 1.25;
  };

  using State = Game::State;

  MCTS(Config config) : config_(config) {}

  constexpr auto search(std::span<const State> original_states,
                        std::shared_ptr<Model> model, int num_simulations,
                        std::optional<std::mt19937*> noise_gen = std::nullopt)
      -> torch::Tensor {
    torch::NoGradGuard no_grad;

    auto device = model->parameters().begin()->device();
    auto num_games = original_states.size();

    std::vector<NodeId> root_ids;
    std::vector<NodeId> node_ids;

    std::vector<typename Game::State> states;

    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> legal_actions;

    root_ids.reserve(num_games);
    node_ids.reserve(num_games);

    states.reserve(num_games);

    features.reserve(num_games);
    legal_actions.reserve(num_games);

    for (auto& original_state : original_states) {
      root_ids.emplace_back(nodes_.create(original_state.player));
      features.emplace_back(Game::encode_state(original_state));
      legal_actions.emplace_back(Game::legal_actions(original_state));
    }

    auto [_, policy] = model->forward(torch::stack(features).to(device));
    policy = torch::softmax(policy, 1).cpu();

    if (noise_gen) {
      policy = (1 - config_.dirichlet_epsilon) * policy +
               config_.dirichlet_epsilon *
                   gen_exploration_noise(policy.size(0), *noise_gen);
    }

    for (std::size_t i = 0; i < root_ids.size(); i++) {
      auto priors = policy[i] * legal_actions[i];
      priors /= priors.sum(0);
      expand(root_ids[i], original_states[i], priors);
    }

    for (auto _ : std::views::iota(0, num_simulations)) {
      node_ids.clear();
      features.clear();
      states.clear();
      legal_actions.clear();

      for (std::size_t i = 0; i < root_ids.size(); i++) {
        auto root_id = root_ids[i];
        auto state = original_states[i];

        auto node = nodes_.as_ref(root_id);

        while (node->is_expanded()) {
          node = highest_child_score(node.id);
          state = Game::apply_action(state, node->action);
        }

        if (auto outcome = Game::get_outcome(state, node->action)) {
          auto& parent = nodes_.get(node->parent_id);
          backpropagate(node.id, outcome->as_scalar(), parent.player);
        } else {
          node_ids.push_back(node.id);
          states.emplace_back(state);
          features.emplace_back(Game::encode_state(state));
          legal_actions.emplace_back(Game::legal_actions(state));
        }
      }

      auto [wdl, policy] = model->forward(torch::stack(features).to(device));
      policy = torch::softmax(policy, 1).cpu();
      wdl = wdl.cpu();

      for (std::size_t i = 0; i < node_ids.size(); i++) {
        auto node = nodes_.as_ref(node_ids[i]);
        auto priors = policy[i] * legal_actions[i];
        priors /= priors.sum(0);

        expand(node.id, states[i], priors);

        auto value =
            wdl[i][0].template item<float>() - wdl[i][2].template item<float>();
        backpropagate(node.id, value, states[i].player);
      }
    }

    auto child_visits = torch::zeros(
        {static_cast<int>(num_games), Game::ActionSize}, torch::kFloat32);

    for (size_t i = 0; i < root_ids.size(); i += 1) {
      const auto root_id = root_ids[i];

      for (auto child_id : nodes_.get(root_id).children()) {
        auto& child = nodes_.get(child_id);
        child_visits[i][child.action] = auto(child.visits);
      }
    }

    nodes_.clear();

    return child_visits / child_visits.sum(1, /*keep_dims=*/true);
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

  constexpr auto expand(NodeId parent_id, const Game::State& state,
                        torch::Tensor policy) -> void {
    auto parent = nodes_.as_ref(parent_id);
    for (std::size_t action = 0; action < Game::ActionSize; action++) {
      if (auto prior = policy[action].template item<float>(); prior > 0) {
        auto new_state = Game::apply_action(state, action);
        parent.create_child(new_state.player, action, prior);
      }
    }
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

  constexpr auto gen_exploration_noise(int batch_size, std::mt19937* gen)
      -> torch::Tensor {
    auto gamma =
        std::gamma_distribution<float32_t>(config_.dirichlet_alpha, 1.0);

    auto dirichlet_noise = torch::zeros({batch_size, Game::ActionSize});
    for (auto batch = 0; batch < batch_size; batch++) {
      for (auto action = 0; action < Game::ActionSize; action++) {
        dirichlet_noise[batch][action] = gamma(*gen);
      }
    }

    return dirichlet_noise;
  }

 private:
  NodeStorage nodes_;
  Config config_;
};

}  // namespace az
