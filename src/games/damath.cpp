#include <torch/torch.h>

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
  using Action = AZ::Action;
  using Player = AZ::Player;
  using Network = Network;

  static constexpr auto ActionSize = 8 * 8 * 4 * 7;

  struct Board {
    struct Piece {
      std::uint8_t occup : 1;
      std::uint8_t enemy : 1;
      std::uint8_t queen : 1;
      std::uint8_t ngtve : 1;
      std::uint8_t value : 4;
    };

    static constexpr std::array<std::array<const char, 8>, 8> operators{
        {{' ', '+', ' ', '-', ' ', '/', ' ', '*'},
         {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
         {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
         {'*', ' ', '/', ' ', '-', ' ', '+', ' '},
         {' ', '+', ' ', '-', ' ', '/', ' ', '*'},
         {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
         {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
         {'*', ' ', '/', ' ', '-', ' ', '+', ' '}}};

    std::array<std::array<Board::Piece, 8>, 8> pieces{{
        // clang-format off
      {{{0,0,0,0,0},{1,0,0,1,11},{0,0,0,0,0},{1,0,0,0,8},{0,0,0,0,0},{1,0,0,1,5},{0,0,0,0,0},{1,0,0,0,2}}},
      {{{1,0,0,0,0},{0,0,0,0,0},{1,0,0,1,3},{0,0,0,0,0},{1,0,0,0,10},{0,0,0,0,0},{1,0,0,1,7},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{1,0,0,1,9},{0,0,0,0,0},{1,0,0,0,6},{0,0,0,0,0},{1,0,0,1,1},{0,0,0,0,0},{1,0,0,0,4}}},
      {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
      {{{1,1,0,0,4},{0,0,0,0,0},{1,1,0,1,1},{0,0,0,0,0},{1,1,0,0,6},{0,0,0,0,0},{1,1,0,1,9},{0,0,0,0,0}}},
      {{{0,0,0,0,0},{1,1,0,1,7},{0,0,0,0,0},{1,1,0,0,10},{0,0,0,0,0},{1,1,0,1,3},{0,0,0,0,0},{1,1,0,0,0}}},
      {{{1,1,0,0,2},{0,0,0,0,0},{1,1,0,1,5},{0,0,0,0,0},{1,1,0,0,8},{0,0,0,0,0},{1,1,0,1,11},{0,0,0,0,0}}},
    }};  // clang-format on
  };

  struct State {
    Board board = Board{};
    Player player = Player::First;
    std::pair<double, double> scores{0.0, 0.0};
  };

  static constexpr auto flip(const Board& board) -> Board {
    auto new_board = Board{
        .pieces =  // clang-format off
            {{{{board.pieces[7][7], board.pieces[7][6], board.pieces[7][5], board.pieces[7][4], board.pieces[7][3], board.pieces[7][2], board.pieces[7][1], board.pieces[7][0]}},
              {{board.pieces[6][7], board.pieces[6][6], board.pieces[6][5], board.pieces[6][4], board.pieces[6][3], board.pieces[6][2], board.pieces[6][1], board.pieces[6][0]}},
              {{board.pieces[5][7], board.pieces[5][6], board.pieces[5][5], board.pieces[5][4], board.pieces[5][3], board.pieces[5][2], board.pieces[5][1], board.pieces[5][0]}},
              {{board.pieces[4][7], board.pieces[4][6], board.pieces[4][5], board.pieces[4][4], board.pieces[4][3], board.pieces[4][2], board.pieces[4][1], board.pieces[4][0]}},
              {{board.pieces[3][7], board.pieces[3][6], board.pieces[3][5], board.pieces[3][4], board.pieces[3][3], board.pieces[3][2], board.pieces[3][1], board.pieces[3][0]}},
              {{board.pieces[2][7], board.pieces[2][6], board.pieces[2][5], board.pieces[2][4], board.pieces[2][3], board.pieces[2][2], board.pieces[2][1], board.pieces[2][0]}},
              {{board.pieces[1][7], board.pieces[1][6], board.pieces[1][5], board.pieces[1][4], board.pieces[1][3], board.pieces[1][2], board.pieces[1][1], board.pieces[1][0]}},
              {{board.pieces[0][7], board.pieces[0][6], board.pieces[0][5], board.pieces[0][4], board.pieces[0][3], board.pieces[0][2], board.pieces[0][1], board.pieces[0][0]}}
            }},  // clang-format on
    };

    for (auto& row : new_board.pieces)
      for (auto& piece : row)
        if (piece.occup)
          piece.enemy = not piece.enemy;

    return new_board;
  }

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
          static_cast<double>((state.board.pieces[y][x].ngtve ? -1 : 1) *
                              state.board.pieces[y][x].value);
      auto opponent_value = static_cast<double>(
          (state.board.pieces[enemy_y][enemy_x].ngtve ? -1 : 1) *
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
      new_y += distance;

      [&] {
        for (int enemy_y = y + 1; enemy_y < new_y; enemy_y++)
          for (int enemy_x = x - 1; enemy_x > new_x; enemy_x--)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 1) {  // move diagonally to the upper right
      new_x += distance;
      new_y += distance;

      [&] {
        for (int enemy_y = y + 1; enemy_y < new_y; enemy_y++)
          for (int enemy_x = x + 1; enemy_x < new_x; enemy_x++)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 2) {  // move diagonally to the lower left
      new_x -= distance;
      new_y -= distance;

      [&] {
        for (int enemy_y = y - 1; enemy_y > new_y; enemy_y--)
          for (int enemy_x = x - 1; enemy_x > new_x; enemy_x--)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 3) {  // move diagonally to the lower right
      new_x += distance;
      new_y -= distance;

      [&] {
        for (int enemy_y = y - 1; enemy_y > new_y; enemy_y--)
          for (int enemy_x = x + 1; enemy_x < new_x; enemy_x++)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
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

    if (get_eat_actions(new_state, new_x, new_y).size() > 0)
      return new_state;

    new_state.board = flip(new_state.board);
    new_state.player = state.player.next();
    return new_state;
  }

  struct ActionNode {
    int action;
    std::vector<ActionNode> children;

    auto height() const -> int {
      if (children.empty())
        return 1;
      else
        return 1 + std::ranges::max(
                       children | std::views::transform([](const auto& child) {
                         return child.height();
                       }));
    }
  };

  static constexpr auto get_eat_actions(const State& state, int x, int y)
      -> std::vector<ActionNode> {
    auto actions = std::vector<ActionNode>{};

    auto piece = state.board.pieces[y][x];

    if (piece.queen) {
      for (auto distance = 2; x - distance >= 0 and y + distance < 8;
           distance++)
        if (not state.board.pieces[y + distance][x - distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y + enemy][x - enemy].enemy) {
              auto valid = true;

              for (auto between = distance - enemy + 1; between < distance;
                   between++)
                if (state.board.pieces[y + between][x - between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 0) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x - distance, y + distance)});
              }
            }
      for (auto distance = 2; x + distance < 8 and y + distance < 8; distance++)
        if (not state.board.pieces[y + distance][x + distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y + enemy][x + enemy].enemy) {
              auto valid = true;

              for (auto between = distance - enemy + 1; between < distance;
                   between++)
                if (state.board.pieces[y + between][x + between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 1) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x + distance, y + distance)});
              }
            }
      for (auto distance = 2; x - distance >= 0 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x - distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y - enemy][x - enemy].enemy) {
              auto valid = true;

              for (auto between = distance - enemy + 1; between < distance;
                   between++)
                if (state.board.pieces[y - between][x - between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 2) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x - distance, y - distance)});
              }
            }
      for (auto distance = 2; x + distance < 8 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x + distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y - enemy][x + enemy].enemy) {
              auto valid = true;

              for (auto between = distance - enemy + 1; between < distance;
                   between++)
                if (state.board.pieces[y - between][x + between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 3) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x + distance, y - distance)});
              }
            }
    } else {
      if (y + 2 < 8) {
        if (x - 2 >= 0 and not state.board.pieces[y + 2][x - 2].occup and
            state.board.pieces[y + 1][x - 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(
              {action,
               get_eat_actions(apply_action(state, action), x - 2, y + 2)});
        }
        if (x + 2 < 8 and not state.board.pieces[y + 2][x + 2].occup and
            state.board.pieces[y + 1][x + 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(
              {action,
               get_eat_actions(apply_action(state, action), x + 2, y + 2)});
        }
      }
    }

    return actions;
  }

  static constexpr auto get_jump_actions(const State& state, int x, int y)
      -> std::vector<Action> {
    auto actions = std::vector<Action>{};

    auto piece = state.board.pieces[y][x];

    assert(not piece.enemy);

    if (piece.queen) {
      for (auto distance = 1; x - distance >= 0 and y + distance < 8;
           distance++)
        if (not state.board.pieces[y + distance][x - distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y + between][x - between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 0) + (8 * y) + x;
            actions.push_back(action);
          }
        }
      for (auto distance = 1; x + distance < 8 and y + distance < 8; distance++)
        if (not state.board.pieces[y + distance][x + distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y + between][x + between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 1) + (8 * y) + x;
            actions.push_back(action);
          }
        }
      for (auto distance = 1; x - distance >= 0 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x - distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y - between][x - between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 2) + (8 * y) + x;
            actions.push_back(action);
          }
        }
      for (auto distance = 1; x + distance < 8 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x + distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y - between][x + between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 3) + (8 * y) + x;
            actions.push_back(action);
          }
        }
    } else {
      if (y + 1 < 8) {
        if (x - 1 >= 0 and not state.board.pieces[y + 1][x - 1].occup) {
          auto action = (8 * 8 * 4 * 0) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(action);
        }
        if (x + 1 < 8 and not state.board.pieces[y + 1][x + 1].occup) {
          auto action = (8 * 8 * 4 * 0) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    }

    return actions;
  }

  static constexpr auto legal_actions(const State& state) -> torch::Tensor {
    auto legal_actions = std::vector<Action>{};

    auto pieces = std::vector<std::pair<int, int>>{};

    for (int y = 0; y < 8; y++)
      for (int x = 0; x < 8; x++)
        if (state.board.pieces[y][x].occup)
          if (not state.board.pieces[y][x].enemy)
            pieces.push_back({x, y});

    auto eat_actions = std::vector<ActionNode>{};
    for (auto& [x, y] : pieces)
      eat_actions.append_range(get_eat_actions(state, x, y));

    auto heights =
        eat_actions |
        std::views::transform([](const auto& node) { return node.height(); }) |
        std::ranges::to<std::vector>();

    for (auto it = std::max_element(heights.begin(), heights.end());
         it != heights.end(); it = std::find(it + 1, heights.end(), *it))
      legal_actions.push_back(
          eat_actions[std::distance(heights.begin(), it)].action);

    auto legal_actions_tensor = torch::zeros(ActionSize, torch::kFloat32);
    if (not legal_actions.empty()) {
      for (auto& action : legal_actions)
        legal_actions_tensor[action] = 1.0;

      return legal_actions_tensor;
    }

    for (auto& [x, y] : pieces)
      legal_actions.append_range(get_jump_actions(state, x, y));

    for (auto& action : legal_actions)
      legal_actions_tensor[action] = 1.0;

    return legal_actions_tensor;
  }

  static constexpr auto terminal_value(const State& state, Action action)
      -> std::optional<double> {
    if (legal_actions(apply_action(state, action)).sum(0).item<double>() ==
        0.0) {
      auto [first, second] = state.scores;
      if (first > second)
        return {1.0};
      else if (first < second)
        return {-1.0};
      else
        return {0.0};
    } else
      return std::nullopt;
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
    for (int i = 7; i >= 0; i--) {
      for (int j = 0; j < 8; j++) {
        if (board.pieces[i][j].occup) {
          std::print(" {: ^3} ", (board.pieces[i][j].ngtve ? -1 : 1) *
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

  auto get_action = [&](auto action) {
    auto distance = (action / (8 * 8 * 4)) + 1;
    auto direction = (action % (8 * 8 * 4)) / (8 * 8);
    auto y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
    auto x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;
    return std::make_tuple(x, y, direction, distance);
  };

  auto legal_actions = Damath::legal_actions(state).nonzero();
  for (int i = 0; i < legal_actions.size(0); i++) {
    auto action = legal_actions[i].item<int>();
    auto [x, y, direction, distance] = get_action(action);
    std::println("({}, {}) {} {} {}", x, y, direction < 2 ? "up" : "down",
                 distance, direction == 0 or direction == 2 ? "left" : "right");
  }

  state = Damath::apply_action(state, action(3, 2, 1, 2));
  Damath::print(state);

  legal_actions = Damath::legal_actions(state).nonzero();
  for (int i = 0; i < legal_actions.size(0); i++) {
    auto action = legal_actions[i].item<int>();
    auto [x, y, direction, distance] = get_action(action);
    std::println("({}, {}) {} {} {}", x, y, direction < 2 ? "up" : "down",
                 distance, direction == 0 or direction == 2 ? "left" : "right");
  }

  state = Damath::apply_action(state, action(1, 2, 1, 2));
  Damath::print(state);

  legal_actions = Damath::legal_actions(state).nonzero();
  for (int i = 0; i < legal_actions.size(0); i++) {
    auto action = legal_actions[i].item<int>();
    auto [x, y, direction, distance] = get_action(action);
    std::println("({}, {}) {} {} {}", x, y, direction < 2 ? "up" : "down",
                 distance, direction == 0 or direction == 2 ? "left" : "right");
  }

  return 0;
}