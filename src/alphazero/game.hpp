#pragma once

#include <torch/torch.h>

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

}  // namespace az
