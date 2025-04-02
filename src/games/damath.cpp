#include <arm_neon.h>
#include <torch/torch.h>

#include <cstdint>

import std;
import alphazero;

struct Network : torch::nn::Module {
  torch::nn::Linear fc1, fc2, value_head, policy_head;
  torch::nn::BatchNorm1d bn1, bn2;

  Network()
      : fc1(register_module("fc1", torch::nn::Linear(64, 128))),
        fc2(register_module("fc2", torch::nn::Linear(128, 32))),
        value_head(register_module("value", torch::nn::Linear(32, 1))),
        policy_head(
            register_module("policy", torch::nn::Linear(32, 8 * 8 * 4 * 7))),
        bn1(register_module("bn1", torch::nn::BatchNorm1d(32))),
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

struct Damath {
  using Action = AlphaZero::Action;
  using Player = AlphaZero::Player;
  using Network = Network;

  static constexpr auto ActionSize = 8 * 8 * 4 * 7;

  struct Board {
    struct Piece {
      std::uint8_t valid : 1;
      std::uint8_t first : 1;
      std::uint8_t queen : 1;
      std::uint8_t oppos : 1;
      std::uint8_t value : 4;
    };

    static constexpr std::array<std::array<const char, 8>, 8> operators{
        {{'*', ' ', '/', ' ', '-', ' ', '+', ' '},
         {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
         {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
         {' ', '+', ' ', '-', ' ', '/', ' ', '*'},
         {'*', ' ', '/', ' ', '-', ' ', '+', ' '},
         {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
         {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
         {' ', '+', ' ', '-', ' ', '/', ' ', '*'}}};

    // clang-format off
    std::array<std::array<Board::Piece, 8>, 8> pieces{{ 
      {{{1,1,0,0,2},{0,0,0,0,0},{1,1,0,1,5},{0,0,0,0,0},{1,1,0,0,8},{0,0,0,0,0},{1,1,0,1,11},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{1,1,0,1,7},{0,0,0,0,0},{1,1,0,0,10},{0,0,0,0,0},{1,1,0,1,3},{0,0,0,0,0},{1,1,0,0,0}}},
      {{{1,1,0,0,4},{0,0,0,0,0},{1,1,0,1,1},{0,0,0,0,0},{1,1,0,0,6},{0,0,0,0,0},{1,1,0,1,9},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{1,0,0,1,9},{0,0,0,0,0},{1,0,0,0,6},{0,0,0,0,0},{1,0,0,1,1},{0,0,0,0,0},{1,0,0,0,4}}},
      {{{1,0,0,0,0},{0,0,0,0,0},{1,0,0,1,3},{0,0,0,0,0},{1,0,0,0,10},{0,0,0,0,0},{1,0,0,1,7},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{1,0,0,1,11},{0,0,0,0,0},{1,0,0,0,8},{0,0,0,0,0},{1,0,0,1,5},{0,0,0,0,0},{1,0,0,0,2}}},
    }};
    // clang-format on
  };

  struct State {
    Board board = Board{};
    Player player = Player::First;
    std::pair<double, double> scores{0.0, 0.0};
  };

  static constexpr auto initial_state() -> State { return State{}; }

  static constexpr auto apply_action(const State& state, Action action)
      -> State {
    auto distance = (action / (8 * 8 * 4)) + 1;
    auto direction = (action % (8 * 8 * 4)) / (8 * 8);
    auto y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
    auto x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;

    auto new_state = state;
    auto new_x = x;
    auto new_y = y;

    auto eat = [&](auto enemy_x, auto enemy_y) {
      auto op = state.board.operators[new_y][new_x];
      auto& score = state.player.is_first() ? new_state.scores.first
                                            : new_state.scores.second;
      auto player_value =
          static_cast<double>((state.board.pieces[y][x].oppos ? -1 : 1) *
                              state.board.pieces[y][x].value);
      auto opponent_value = static_cast<double>(
          (state.board.pieces[enemy_y][enemy_x].oppos ? -1 : 1) *
          state.board.pieces[enemy_y][enemy_x].value);
      if (op == '+') {
        score += (player_value + opponent_value);
      } else if (op == '-') {
        score += (player_value - opponent_value);
      } else if (op == '*') {
        score += (player_value * opponent_value);
      } else if (op == '/') {
        score += (player_value / opponent_value);
      }
      new_state.board.pieces[enemy_y][enemy_x] = {0, 0, 0, 0, 0};
    };

    if (direction == 0) {  // move diagonally to the upper left
      new_x -= distance;
      new_y -= distance;

      [&] {
        for (int enemy_y = y - 1; enemy_y > new_y; enemy_y--)
          for (int enemy_x = x - 1; enemy_x > new_x; enemy_x--)
            if (state.board.pieces[enemy_y][enemy_x].valid) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 1) {  // move diagonally to the upper right
      new_x += distance;
      new_y -= distance;

      [&] {
        for (int enemy_y = y - 1; enemy_y > new_y; enemy_y--)
          for (int enemy_x = x + 1; enemy_x < new_x; enemy_x++)
            if (state.board.pieces[enemy_y][enemy_x].valid) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 2) {  // move diagonally to the lower left
      new_x -= distance;
      new_y += distance;

      [&] {
        for (int enemy_y = y + 1; enemy_y < new_y; enemy_y++)
          for (int enemy_x = x - 1; enemy_x > new_x; enemy_x--)
            if (state.board.pieces[enemy_y][enemy_x].valid) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 3) {  // move diagonally to the lower right
      new_x += distance;
      new_y += distance;

      [&] {
        for (int enemy_y = y + 1; enemy_y < new_y; enemy_y++)
          for (int enemy_x = x + 1; enemy_x < new_x; enemy_x++)
            if (state.board.pieces[enemy_y][enemy_x].valid) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    }

    new_state.board.pieces[y][x] = {0, 0, 0, 0, 0};
    new_state.board.pieces[new_y][new_x] = state.board.pieces[y][x];

    if (not new_state.board.pieces[new_y][new_x].queen and
        (new_y == 0 or new_y == 7))
      new_state.board.pieces[new_y][new_x].queen = true;

    // if (checkMultipleJumps(new_state, new_x, new_y))
    //   // if the player can jump again, do not change the player
    //   return new_state;

    new_state.player = state.player.next();
    return new_state;
  }

  static constexpr auto legal_actions(const State& _) -> torch::Tensor {
    auto legal_actions = torch::zeros({8, 8, 4, 7}, torch::kFloat32);

    // // check for multiple eating moves
    // for (std::size_t y = 0; y < 8; y++)
    //   for (std::size_t x = 0; x < 8; x++)
    //     if (state.board.pieces[y][x].valid) {
    //       // only check for eats if the piece is valid
    //       for (std::size_t direction = 0; direction < 4; direction++)
    //         for (std::size_t distance = 2; distance <= 7; distance++) {
    //           legal_actions[y][x][direction][distance] = [&] {
    //             if (not state.board.pieces[y][x].queen and distance > 2)
    //               // only queens can move more than 2 spaces
    //               return 0.0;

    //             auto new_x = x;
    //             auto new_y = y;

    //             if (direction == 0) {
    //               // move diagonally to the upper left
    //               new_x -= distance;
    //               new_y -= distance;
    //             } else if (direction == 1) {
    //               // move diagonally to the upper right
    //               new_x += distance;
    //               new_y -= distance;
    //             } else if (direction == 2) {
    //               // move diagonally to the lower left
    //               new_x -= distance;
    //               new_y += distance;
    //             } else if (direction == 3) {
    //               // move diagonally to the lower right
    //               new_x += distance;
    //               new_y += distance;
    //             }

    //             if (new_x < 0 or new_x > 7 or new_y < 0 or
    //                 new_y > 7)  // out of bounds
    //               return 0.0;

    //             if (state.board.pieces[new_y][new_x].valid)  // already
    //             occupied
    //               return 0.0;

    //             return 1.0;
    //           }();
    //         }
    //     }

    // for (std::size_t y = 0; y < 8; y++)
    //   for (std::size_t x = 0; x < 8; x++)
    //     if (state.board.pieces[y][x].valid) {
    //       for (std::size_t direction = 0; direction < 4; direction++) {
    //         auto jumps = std::vector<Action>{};
    //         legal_actions[y][x][direction][1] = [&] {
    //           auto enemy_x = x;
    //           auto enemy_y = y;

    //           auto new_x = x;
    //           auto new_y = y;

    //           if (direction == 0) {
    //             // move diagonally to the upper left
    //             enemy_x -= 1;
    //             enemy_y -= 1;
    //             new_x -= 2;
    //             new_y -= 2;
    //           } else if (direction == 1) {
    //             // move diagonally to the upper right
    //             enemy_x += 1;
    //             enemy_y -= 1;
    //             new_x += 2;
    //             new_y -= 2;
    //           }

    //           if (new_x < 0 or new_x > 7 or new_y < 0 or
    //               new_y > 7)  // out of bounds
    //             return 0.0;

    //           if (state.board.pieces[new_y][new_x].valid)  // already
    //           occupied
    //             return 0.0;

    //           if (not state.board.pieces[enemy_y][enemy_x].enemy)
    //             return 0.0;

    //           return 1.0;
    //         }();
    //       }

    //       for (std::size_t direction = 0; direction < 4; direction++) {
    //         legal_actions[y][x][direction][1] = [&] {
    //           auto new_x = x;
    //           auto new_y = y;

    //           if (direction == 0) {
    //             // move diagonally to the upper left
    //             new_x -= 1;
    //             new_y -= 1;
    //           } else if (direction == 1) {
    //             // move diagonally to the upper right
    //             new_x += 1;
    //             new_y -= 1;
    //           }

    //           if (new_x < 0 or new_x > 7 or new_y < 0 or
    //               new_y > 7)  // out of bounds
    //             return 0.0;

    //           if (state.board.pieces[new_y][new_x].valid)  // already
    //           occupied
    //             return 0.0;

    //           return 1.0;
    //         }();
    //       }
    //     }

    return legal_actions;
  }

  static constexpr auto terminal_value(const State& state, Action)
      -> std::optional<double> {
    if (legal_actions(state).sum(0).item<double>() == 0.0) {
      auto [first, second] = state.scores;
      if (first > second)
        return {1.0};
      else if (first < second)
        return {-1.0};
      else
        return {0.0};
    } else
      return {};
  }

  // static constexpr auto encode_state(const State& state) -> torch::Tensor {
  //   auto encoded_state = torch::zeros(ActionSize, torch::kFloat32);
  //   auto flip = state.player.is_first() ? 1 : -1;
  //   for (std::size_t i = 0; i < state.data.size(); i++)
  //     encoded_state[i] = state.data[i] * flip;

  //   return encoded_state;
  // }

  static constexpr auto print(const State& state) -> void {
    auto& board = state.board;
    auto [first, second] = state.scores;
    std::print("Score: {:.5f} {:.5f}\n", first, second);
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        if (board.pieces[i][j].valid == 1) {
          std::print(" {: ^3} ", (board.pieces[i][j].oppos ? -1 : 1) *
                                     board.pieces[i][j].value);
        } else {
          std::print(" {: ^3} ", board.operators[i][j]);
        }
      }
      std::println();
    }
  }
};

// static_assert(AlphaZero::Concepts::Game<Damath>);

// struct Agent {
//   static constexpr auto player = AlphaZero::Player::First;

//   Agent(std::shared_ptr<Network> model) : model(model) {}

//   auto on_move(const Damath::State& state) -> AlphaZero::Action {
//     Damath::print(state);

//     std::cout << Damath::legal_actions(state).nonzero() << '\n';

//     int input = 0;
//     std::cout << "Enter action: ";
//     std::cin >> input;
//     return static_cast<AlphaZero::Action>(input);
//   }

//   auto on_model_move(const Damath::State& state, torch::Tensor probs,
//                      AlphaZero::Action _) -> void {
//     auto feature = torch::unsqueeze(Damath::encode_state(state), 0);

//     auto [value, policy] = model->forward(feature);
//     policy = torch::softmax(torch::squeeze(policy, 0), -1);
//     policy *= Damath::legal_actions(state);
//     policy /= policy.sum();

//     std::cout << "Policy: " << policy << "\n";
//     std::cout << "MCTS: " << probs << "\n";
//     std::cout << "Value: " << value << "\n";
//   }

//   auto on_game_end(const Damath::State& state, AlphaZero::GameResult
//   result)
//       -> void {
//     auto new_state = state;

//     // flip the player before printing it>
//     new_state.player = player;
//     Damath::print(new_state);

//     switch (result) {
//       case AlphaZero::GameResult::Win:
//         std::println("You won!");
//         break;
//       case AlphaZero::GameResult::Lost:
//         std::println("You lost!");
//         break;
//       case AlphaZero::GameResult::Draw:
//         std::println("Draw!");
//         break;
//     }
//   }

//   std::shared_ptr<Network> model;
// };

// static_assert(AlphaZero::Concepts::Agent<Agent, Damath>);

// auto main() -> int {
//   torch::DeviceGuard device_guard(torch::kCPU);
//   auto config = AlphaZero::Config{
//       .num_iterations = 1,
//       .num_simulations = 60,
//       .num_self_play_iterations = 500,
//       .num_actors = 6,
//       .device = torch::kCPU,
//   };

//   auto model = std::make_shared<Network>();
//   auto optimizer = std::make_shared<torch::optim::Adam>(
//       model->parameters(), torch::optim::AdamOptions(0.001));

//   auto rng = std::random_device{};

//   auto alpha_zero = AlphaZero::AlphaZero<Damath>{
//       config,
//       model,
//       optimizer,
//       rng,
//   };

//   alpha_zero.learn();

//   auto arena = AlphaZero::Arena<Damath>(config);
//   arena.play_with_model(model, 1000, Agent{model});

//   return 0;
// }

auto main() -> int {
  auto state = Damath::initial_state();
  Damath::print(state);

  auto action = [&](auto x, auto y, auto direction, auto distance) {
    std::println("Player {} moves ({},{}) {} {} block to the {}!",
                 state.player.is_first() ? 1 : 2, x, y,
                 direction < 2 ? "up" : "down", distance,
                 direction == 0 or direction == 2 ? "left" : "right");
    return (8 * 8 * 4 * (distance - 1)) + (8 * 8 * direction) + (8 * y) + (x);
  };
  state = Damath::apply_action(state, action(1, 5, 1, 2));
  Damath::print(state);

  state = Damath::apply_action(state, action(4, 2, 2, 2));
  Damath::print(state);

  return 0;
}