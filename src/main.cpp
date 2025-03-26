#include <torch/torch.h>

import std;
import damathzero;

auto print_board(const DamathZero::Board& board, DamathZero::Player player) {
  for (int i = 0; i < 9; i++) {
    if (i % 3 == 0) {
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

  auto damath_zero =
      DamathZero::DamathZero{{}, model, optimizer, random_device};
  damath_zero.learn();

  auto mcts = DamathZero::MCTS{{.NumSimulations = 1000}};
  auto board = DamathZero::Game::initial_board();
  auto player = DamathZero::Player{1};

  auto action = -1;

  while (true) {
    if (player == 1) {
      print_board(board, player);

      std::println("{}", DamathZero::Game::legal_actions(board));

      int input = 0;
      std::cout << "Enter action: ";
      std::cin >> input;

      action = static_cast<DamathZero::Action>(input);
    } else {
      model->eval();
      torch::NoGradGuard no_grad;

      auto neutral_state = DamathZero::Game::change_perspective(board, player);
      auto probs = mcts.search(neutral_state, model);

      auto [value, policy] = model->forward(
          torch::unsqueeze(torch::tensor(neutral_state, torch::kFloat32), 0));
      policy = torch::squeeze(policy, 0);
      policy = policy.index(
          {torch::tensor(DamathZero::Game::legal_actions(neutral_state))});
      policy /= policy.sum();

      std::cout << "Policy: " << policy << "\n";
      std::cout << "MCTS: " << probs << "\n";
      std::cout << "Value: " << value << "\n";

      action = torch::argmax(policy).item<int>();

      while (not std::ranges::contains(DamathZero::Game::legal_actions(board),
                                       action)) {
        policy[action] = 0;
        action = torch::argmax(policy).item<int>();
      }
    }

    auto [new_board, new_player] =
        DamathZero::Game::apply_action(board, action, player);

    board = new_board;

    auto [value, is_terminal] =
        DamathZero::Game::get_value_and_terminated(board, action);

    if (is_terminal) {
      print_board(board, player);
      if (value * player == 1) {
        std::println("You won!");
      } else if (value * player == -1) {
        std::println("You lost!");
      } else {
        std::println("Draw!");
      }
      break;
    }

    player = new_player;
  }

  return 0;
}
