module;

#include <torch/torch.h>

export module dz:agent;

import :game;
import :model;

import az;
import std;

namespace dz {

export struct Agent {
  auto on_move(const dz::Game::State& state) -> az::Action {
    dz::Game::print(state);

    auto get_action = [&](auto action) {
      auto distance = (action / (8 * 8 * 4)) + 1;
      auto direction = (action % (8 * 8 * 4)) / (8 * 8);
      auto y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
      auto x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;
      return std::make_tuple(x, y, direction, distance);
    };

    auto legal_actions = dz::Game::legal_actions(state).nonzero();
    for (int i = 0; i < legal_actions.size(0); i++) {
      auto action = legal_actions[i].item<int>();
      auto [x, y, direction, distance] = get_action(action);
      std::println("[{}] ({}, {}) {} {} {}", action, x, y,
                   direction < 2 ? "up" : "down", distance,
                   direction == 0 or direction == 2 ? "left" : "right");
    }

    int input = 0;
    std::cout << "Enter action: ";
    std::cin >> input;
    return static_cast<az::Action>(input);
  }

  auto on_model_move(const dz::Game::State& state, torch::Tensor probs,
                     az::Action) -> void {
    dz::Game::print(state);
    auto feature = torch::unsqueeze(dz::Game::encode_state(state), 0);

    auto legal_actions = dz::Game::legal_actions(state);

    auto [wdl, policy] = model->forward(feature);
    policy = torch::softmax(torch::squeeze(policy, 0), -1);
    policy *= legal_actions;
    policy /= policy.sum();

    std::cout << "Legal actions: " << legal_actions.nonzero().transpose(0, 1)
              << "\n";
    std::cout << "Policy: "
              << policy.index({legal_actions.nonzero()}).transpose(0, 1)
              << "\n";
    std::cout << "MCTS: "
              << probs.index({legal_actions.nonzero()}).transpose(0, 1) << "\n";
    std::cout << "Win-Draw-Loss: " << wdl << "\n";
  }

  auto on_game_end(const dz::Game::State& state, az::GameOutcome result)
      -> void {
    auto new_state = state;

    // flip the player before printing it>
    new_state.player = player;
    dz::Game::print(new_state);

    if (result == az::GameOutcome::Win) {
      std::println("You won!");
    } else if (result == az::GameOutcome::Loss) {
      std::println("You lost!");
    } else {
      std::println("Draw!");
    }
  }

  std::shared_ptr<Model> model;
  static constexpr auto player = az::Player::First;
};

};  // namespace dz

static_assert(az::concepts::Agent<dz::Agent, dz::Game>);