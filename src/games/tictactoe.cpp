#include <torch/torch.h>

import std;
import damathzero;

struct Network : torch::nn::Module {
  torch::nn::Linear fc1, fc2, value_head, policy_head;
  torch::nn::BatchNorm1d bn1, bn2;

  Network()
      : fc1(register_module("fc1", torch::nn::Linear(9, 64))),
        fc2(register_module("fc2", torch::nn::Linear(64, 32))),
        value_head(register_module("value", torch::nn::Linear(32, 1))),
        policy_head(register_module("policy", torch::nn::Linear(32, 9))),
        bn1(register_module("bn1", torch::nn::BatchNorm1d(64))),
        bn2(register_module("bn2", torch::nn::BatchNorm1d(32))) {}

  auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(bn1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(bn2->forward(x));

    auto value = torch::tanh(value_head->forward(x));
    auto policy = policy_head->forward(x);

    return {value, policy};
  }
};

struct TicTacToe {
  using Action = DamathZero::Action;
  using Player = DamathZero::Player;
  using Network = Network;

  static constexpr auto ActionSize = 9;

  struct State {
    std::vector<int> data;
    Player player;
  };

  static constexpr auto initial_state() -> State {
    return State(std::vector<int>(9, 0), Player::First);
  }

  static constexpr auto apply_action(const State& state, Action action)
      -> State {
    auto new_state = state;
    new_state.data[action] = state.player.is_first() ? 1 : -1;
    new_state.player = state.player.next();
    return new_state;
  }

  static constexpr auto legal_actions(const State& board) -> torch::Tensor {
    auto legal_actions = torch::zeros(ActionSize, torch::kFloat32);

    for (std::size_t i = 0; i < board.data.size(); i++)
      if (board.data[i] == 0)
        legal_actions[i] = 1.0;

    return legal_actions;
  }

  static constexpr auto check_win(const State& board, Action action) -> bool {
    if (action < 0)
      return false;

    auto piece = board.data[action];

    return (board.data[0] == piece and board.data[1] == piece and
            board.data[2] == piece) or
           (board.data[3] == piece and board.data[4] == piece and
            board.data[5] == piece) or
           (board.data[6] == piece and board.data[7] == piece and
            board.data[8] == piece) or
           (board.data[0] == piece and board.data[3] == piece and
            board.data[6] == piece) or
           (board.data[1] == piece and board.data[4] == piece and
            board.data[7] == piece) or
           (board.data[2] == piece and board.data[5] == piece and
            board.data[8] == piece) or
           (board.data[0] == piece and board.data[4] == piece and
            board.data[8] == piece) or
           (board.data[2] == piece and board.data[4] == piece and
            board.data[6] == piece);
  }

  static constexpr auto terminal_value(const State& state, Action action)
      -> std::optional<double> {
    if (check_win(state, action))
      return {1.0};
    else if (legal_actions(state).sum(0).item<double>() == 0.0)
      return {0.0};
    else
      return {};
  }

  static constexpr auto encode_state(const State& state) -> torch::Tensor {
    auto encoded_state = torch::zeros(ActionSize, torch::kFloat32);
    auto flip = state.player.is_first() ? 1 : -1;
    for (std::size_t i = 0; i < state.data.size(); i++)
      encoded_state[i] = state.data[i] * flip;

    return encoded_state;
  }
};

static_assert(DamathZero::Concepts::Game<TicTacToe>);

auto print_board(const TicTacToe::State& state) {
  auto& [board, player] = state;
  auto flip = player.is_first() ? 1 : -1;
  for (int i = 0; i < 9; i++) {
    if (i % 3 == 0) {
      std::cout << "\n";
    }
    std::cout << board[i] * flip << " ";
  }
  std::cout << "\n";
}

auto main() -> int {
  torch::DeviceGuard device_guard(torch::kCPU);
  auto config = DamathZero::Config{
      .num_iterations = 1,
      .num_simulations = 60,
      .device = torch::kCPU,
  };

  auto model = std::make_shared<Network>();
  auto optimizer = std::make_shared<torch::optim::Adam>(
      model->parameters(), torch::optim::AdamOptions(0.001));

  auto rng = std::random_device{};

  auto alpha_zero = DamathZero::AlphaZero<TicTacToe>{
      config,
      model,
      optimizer,
      rng,
  };

  alpha_zero.learn();

  auto state = TicTacToe::initial_state();
  auto human_player = DamathZero::Player::First;

  auto action = -1;

  while (true) {
    if (state.player == human_player) {
      print_board(state);

      std::cout << TicTacToe::legal_actions(state).nonzero() << '\n';

      int input = 0;
      std::cout << "Enter action: ";
      std::cin >> input;

      action = static_cast<DamathZero::Action>(input);
    } else {
      auto probs = alpha_zero.search(state, 1000);

      action = torch::argmax(probs).item<DamathZero::Action>();

      auto feature = torch::unsqueeze(TicTacToe::encode_state(state), 0);

      auto [value, policy] = model->forward(feature);
      policy = torch::softmax(torch::squeeze(policy, 0), -1);
      policy *= TicTacToe::legal_actions(state);
      policy /= policy.sum();

      std::cout << "Policy: " << policy << "\n";
      std::cout << "MCTS: " << probs << "\n";
      std::cout << "Value: " << value << "\n";
    }

    auto new_state = TicTacToe::apply_action(state, action);
    auto terminal_value = TicTacToe::terminal_value(new_state, action);

    if (terminal_value.has_value()) {
      auto value = *terminal_value;
      value = state.player == human_player ? value : -value;

      // Flip the end state from the perspective of the human player before
      // printing it.
      new_state.player = human_player;
      print_board(new_state);

      if (value == 1) {
        std::println("You won!");
      } else if (value == -1) {
        std::println("You lost!");
      } else {
        std::println("Draw!");
      }
      break;
    }

    state = new_state;
  }

  return 0;
}
