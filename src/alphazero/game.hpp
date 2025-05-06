#pragma once

#include <torch/torch.h>

#include <ranges>

namespace az {

using Action = int;

class GameOutcome;
class Player;

namespace concepts {

template <typename G>
concept Game = requires(const G::State& state, Action action) {
  { state.player } -> std::same_as<const Player&>;

  { G::ActionSize } -> std::same_as<const int&>;

  { G::initial_state() } -> std::same_as<typename G::State>;

  { G::apply_action(state, action) } -> std::same_as<typename G::State>;

  { G::get_outcome(state, action) } -> std::same_as<std::optional<GameOutcome>>;

  { G::legal_actions(state) } -> std::same_as<torch::Tensor>;

  { G::encode_state(state) } -> std::same_as<torch::Tensor>;
};

}  // namespace concepts

class Player {
 public:
  static const Player First;
  static const Player Second;

  constexpr auto next() const -> Player { return is_first_ ? Second : First; }
  constexpr auto operator==(const Player other) const -> bool {
    return other.is_first_ == is_first_;
  }

  constexpr auto is_first() const -> bool { return is_first_; }
  constexpr auto is_second() const -> bool { return not is_first_; }

 private:
  constexpr explicit Player(bool is_first) : is_first_(is_first) {}
  bool is_first_;
};

inline constexpr Player Player::First = Player(true);
inline constexpr Player Player::Second = Player(false);

class GameOutcome {
 public:
  static const GameOutcome Win;
  static const GameOutcome Loss;
  static const GameOutcome Draw;

  constexpr auto as_tensor() const -> torch::Tensor {
    if (value_ == 1) {
      return torch::tensor({1, 0, 0}, torch::kFloat32);
    } else if (value_ == 0) {
      return torch::tensor({0, 1, 0}, torch::kFloat32);
    } else if (value_ == -1) {
      return torch::tensor({0, 0, 1}, torch::kFloat32);
    }
    std::unreachable();
  }

  constexpr auto flip() const -> GameOutcome { return GameOutcome(-value_); }

  constexpr auto as_scalar() const -> float64_t {
    return static_cast<float64_t>(value_);
  }

  constexpr auto operator==(GameOutcome other) const -> bool {
    return value_ == other.value_;
  }

 private:
  constexpr explicit GameOutcome(int8_t value) : value_(value) {}
  int8_t value_;
};

inline constexpr GameOutcome GameOutcome::Win = GameOutcome(1);
inline constexpr GameOutcome GameOutcome::Loss = GameOutcome(-1);
inline constexpr GameOutcome GameOutcome::Draw = GameOutcome(0);

template <concepts::Game Game>
struct ParallelGames {
  using State = Game::State;

  std::vector<State> states;
  std::vector<int32_t> non_terminal_state_indices;

  std::function<void(size_t, GameOutcome, Player)> on_game_end;
  std::function<void(size_t, State, torch::Tensor)> on_game_move;

  ParallelGames(int32_t num_parallel_games,
      std::function<void(int32_t, GameOutcome, Player)> on_game_end,
      std::function<void(int32_t, State, torch::Tensor)> on_game_move = [](auto, auto, auto){}) :  on_game_end(on_game_end), on_game_move(on_game_move) {

    states = std::views::iota(0, num_parallel_games)
            | std::views::transform([](auto _) {return Game::initial_state(); })
            | std::ranges::to<std::vector>();

    non_terminal_state_indices =
        std::views::iota(0, num_parallel_games) | std::ranges::to<std::vector>();
  }

  auto all_terminated() const -> bool {
    return non_terminal_state_indices.size() == 0;
  }

  auto get_non_terminal_states() const -> std::vector<State> {
    return non_terminal_state_indices |
           std::views::transform([this](auto i) { return states[i]; }) |
           std::ranges::to<std::vector>();
  }

  auto apply_to_non_terminal_states(torch::Tensor action_probs) -> void {
    // action probs has a batch
    assert(action_probs.sizes().size() == 2);
    assert(non_terminal_state_indices.size() == action_probs.size(0));

    auto i = 0;
    auto actions = torch::multinomial(action_probs, 1).squeeze(1);
    auto to_erase = std::vector<int32_t>();
    for (auto game_index : non_terminal_state_indices) {
      on_game_move(game_index, states[game_index], action_probs);
      const auto action = actions[i].template item<Action>();
      const auto new_state = Game::apply_action(states[game_index], action);

      if (const auto outcome = Game::get_outcome(new_state, action)) {
        on_game_end(game_index, *outcome, states[game_index].player);
        to_erase.push_back(game_index);
      }

      states[game_index] = std::move(new_state);
      i += 1;
    }

    for (auto i : to_erase) {
      non_terminal_state_indices.erase(non_terminal_state_indices.begin() + i);
    }
  }
};

}  // namespace az
