#include <torch/torch.h>

import std;
import damathzero;

auto print_board(const DamathZero::Board& board, DamathZero::Player player) {
    for (int i = 0; i < 9; i++) {
        if (i % 3 ==0) {
            std::cout << "\n";
        }
        std::cout << player * board[i] << " ";
    }
    std::cout << "\n";
}

auto main() -> int {
  auto model = std::make_shared<DamathZero::Network>();
  auto optimizer = std::make_shared<torch::optim::Adam>(
      model->parameters(), torch::optim::AdamOptions(0.001));

  auto random_device = std::random_device{};

  auto damath_zero = DamathZero::DamathZero{model, optimizer, random_device};
  damath_zero.learn();

  auto mcts = DamathZero::MCTS{};
  auto board = DamathZero::Game::initial_board();
  auto player = DamathZero::Player{1};

  auto human_player = -1;
  auto computer_player = 1;
  auto action = -1;

  while (true) {

    auto [value, is_terminal] = DamathZero::Game::get_value_and_terminated(board, action);

    if (is_terminal) {
        print_board(board, human_player);
        if (value == 1) {
            std::println("You won!");
        } else if (value == -1) {
            std::println("You lost!");
        } else {
            std::println("Draw!");
        }
        break;
    }

    if (player == human_player) {
        print_board(board, human_player);

        std::println("{}", DamathZero::Game::legal_actions(board));

        int input = 0;
        std::cout << "Enter action: ";
        std::cin >> input;

        action = static_cast<DamathZero::Action>(input);

        std::tie(board, player) = DamathZero::Game::apply_action(board, action, player);
    } else {
        auto neutral_state = DamathZero::Game::change_perspective(board, computer_player);
        auto probs = mcts.search(neutral_state, model);
        action = torch::argmax(probs).item<int>();

        std::tie(board, player) = DamathZero::Game::apply_action(board, action, player);
    }
  }


  // while (not terminal) {
  //   node = mcts.search(node.id, model);
  //   nodes->detach(node.id);


  //   std::tie(value, terminal) =
  //       DamathZero::Game::get_value_and_terminated(node->board, node->action);

  //   if (not terminal) {
  //     for (int i = 0; i < 9; i++) {
  //       if (i % 3 == 0) {
  //         std::cout << "\n";
  //       }
  //       std::cout << node->board[i] << " ";
  //     }

  //     std::tie(value, terminal) =
  //         DamathZero::Game::get_value_and_terminated(node->board, node->action);
  //   }
  // }

  // return 0;
}
