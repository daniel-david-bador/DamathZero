#include <torch/torch.h>

import std;
import damathzero;

auto main() -> int {
  auto model = std::make_shared<DamathZero::Network>();
  auto optimizer = std::make_shared<torch::optim::Adam>(
      model->parameters(), torch::optim::AdamOptions(0.001));

  auto random_device = std::random_device{};

  auto damath_zero = DamathZero::DamathZero{model, optimizer, random_device};
  damath_zero.learn();

  auto nodes = std::make_shared<DamathZero::NodeStorage>();
  auto mcts = DamathZero::MCTS{nodes};
  auto node = nodes->as_ref(nodes->create());

  auto [value, terminal] =
      DamathZero::Game::get_value_and_terminated(node->board, node->action);

  while (not terminal) {
    node = mcts.search(node.id, model);
    nodes->detach(node.id);

    for (int i = 0; i < 9; i++) {
      if (i % 3 == 0) {
        std::cout << "\n";
      }
      std::cout << node->board[i] << " ";
    }
    std::cout << "\n";

    std::tie(value, terminal) =
        DamathZero::Game::get_value_and_terminated(node->board, node->action);

    if (not terminal) {
      for (int i = 0; i < 9; i++) {
        if (i % 3 == 0) {
          std::cout << "\n";
        }
        std::cout << node->board[i] << " ";
      }
      std::println("{}", DamathZero::Game::legal_actions(node->board));

      int input = 0;
      std::cout << "Enter action: ";
      std::cin >> input;

      auto action = static_cast<DamathZero::Action>(input);
      auto child_board = DamathZero::Game::apply_action(node->board, action, 1);
      child_board = DamathZero::Game::change_perpective(child_board, -1);
      node =
          nodes->create_child(node.id, child_board, -1, action, 0.0, node.id);

      std::tie(value, terminal) =
          DamathZero::Game::get_value_and_terminated(node->board, node->action);
    }
  }

  return 0;
}
