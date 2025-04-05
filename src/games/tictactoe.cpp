#include <torch/torch.h>

import std;
import alphazero;

struct Network : public torch::nn::Module {
  Network()
      : fc1(register_module("fc1", torch::nn::Linear(9, 64))),
        fc2(register_module("fc2", torch::nn::Linear(64, 32))),
        wdl_head(register_module("wdl", torch::nn::Linear(32, 3))),
        policy_head(register_module("policy", torch::nn::Linear(32, 9))),
        bn1(register_module("bn1", torch::nn::BatchNorm1d(64))),
        bn2(register_module("bn2", torch::nn::BatchNorm1d(32))) {}

  auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(bn1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(bn2->forward(x));

    auto wdl = torch::softmax(wdl_head->forward(x), -1);
    auto policy = policy_head->forward(x);

    return {wdl, policy};
  }

  torch::nn::Linear fc1, fc2, wdl_head, policy_head;
  torch::nn::BatchNorm1d bn1, bn2;
};

static_assert(AZ::Concepts::Network<Network>);

struct TicTacToe {
  static constexpr auto ActionSize = 9;

  using Board = std::vector<int>;

  struct State {
    Board board;
    AZ::Player player;
  };

  static constexpr auto initial_state() -> State {
    return State(std::vector<int>(9, 0), AZ::Player::First);
  }

  static constexpr auto apply_action(const State& state, AZ::Action action)
      -> State {
    auto new_state = state;
    new_state.board[action] = state.player.is_first() ? 1 : -1;
    new_state.player = state.player.next();
    return new_state;
  }

  static constexpr auto legal_actions(const State& state) -> torch::Tensor {
    auto legal_actions = torch::zeros(ActionSize, torch::kFloat32);

    for (std::size_t i = 0; i < state.board.size(); i++)
      if (state.board[i] == 0)
        legal_actions[i] = 1.0;

    return legal_actions;
  }

  static constexpr auto check_win(const State& state, AZ::Action action)
      -> bool {
    if (action < 0)
      return false;

    auto piece = state.board[action];

    return (state.board[0] == piece and state.board[1] == piece and
            state.board[2] == piece) or
           (state.board[3] == piece and state.board[4] == piece and
            state.board[5] == piece) or
           (state.board[6] == piece and state.board[7] == piece and
            state.board[8] == piece) or
           (state.board[0] == piece and state.board[3] == piece and
            state.board[6] == piece) or
           (state.board[1] == piece and state.board[4] == piece and
            state.board[7] == piece) or
           (state.board[2] == piece and state.board[5] == piece and
            state.board[8] == piece) or
           (state.board[0] == piece and state.board[4] == piece and
            state.board[8] == piece) or
           (state.board[2] == piece and state.board[4] == piece and
            state.board[6] == piece);
  }

  static constexpr auto get_outcome(const State& state, AZ::Action action)
      -> std::optional<AZ::GameOutcome> {
    if (check_win(state, action))
      return AZ::GameOutcome::Win;
    else if (legal_actions(state).sum(0).item<double>() == 0.0)
      return AZ::GameOutcome::Draw;
    else
      return {};
  }

  static constexpr auto encode_state(const State& state) -> torch::Tensor {
    auto encoded_state = torch::zeros(ActionSize, torch::kFloat32);
    auto flip = state.player.is_first() ? 1 : -1;
    for (std::size_t i = 0; i < state.board.size(); i++)
      encoded_state[i] = state.board[i] * flip;

    return encoded_state;
  }

  static constexpr auto print(const State& state) -> void {
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
};

static_assert(AZ::Concepts::Game<TicTacToe>);

struct Agent {
  static constexpr auto player = AZ::Player::First;

  Agent(std::shared_ptr<Network> model) : model(model) {}

  auto on_move(const TicTacToe::State& state) -> AZ::Action {
    TicTacToe::print(state);

    std::cout << TicTacToe::legal_actions(state).nonzero().transpose(0, 1)
              << '\n';

    int input = 0;
    std::cout << "Enter action: ";
    std::cin >> input;
    return static_cast<AZ::Action>(input);
  }

  auto on_model_move(const TicTacToe::State& state, torch::Tensor probs,
                     AZ::Action _) -> void {
    auto feature = torch::unsqueeze(TicTacToe::encode_state(state), 0);

    auto [value, policy] = model->forward(feature);
    policy = torch::softmax(torch::squeeze(policy, 0), -1);
    policy *= TicTacToe::legal_actions(state);
    policy /= policy.sum();

    std::cout << "Policy:\n" << policy.reshape({3, 3}) << "\n";
    std::cout << "MCTS:\n" << probs.reshape({3, 3}) << "\n";
    std::cout << "Value: " << value << "\n";
  }

  auto on_game_end(const TicTacToe::State& state, AZ::GameOutcome result)
      -> void {
    auto new_state = state;

    // flip the player before printing it>
    new_state.player = player;
    TicTacToe::print(new_state);

    if (result == AZ::GameOutcome::Win) {
      std::println("You won!");
    } else if (result == AZ::GameOutcome::Loss) {
      std::println("You lost!");
    } else {
      std::println("Draw!");
    }
  }

  std::shared_ptr<Network> model;
};

static_assert(AZ::Concepts::Agent<Agent, TicTacToe>);

auto main() -> int {
  auto config = AZ::Config{
      .num_iterations = 1,
      .num_simulations = 60,
      .num_self_play_iterations_per_actor = 100,
      .num_actors = 5,
      .device = torch::kCPU,
  };

  auto gen = std::mt19937{};

  auto alpha_zero = AZ::AlphaZero<TicTacToe, Network>{
      config,
      gen,
  };

  auto model = alpha_zero.learn();

  auto arena = AZ::Arena<TicTacToe, Network>(config);
  arena.play_with_model(model, /*num_simulations=*/1000, Agent{model});

  return 0;
}
