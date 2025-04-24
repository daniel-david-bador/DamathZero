module;

#include <torch/torch.h>

export module az:game;

import std;

namespace az {

export using Action = int;

export class GameOutcome;
export class Player;

namespace concepts {

export template <typename G>
concept Game = requires(const G::State& state, Action action) {
  typename G::State;

  std::same_as<decltype(state.player), Player>;
  std::same_as<decltype(G::ActionSize), int>;

  { G::initial_state() } -> std::same_as<typename G::State>;

  { G::apply_action(state, action) } -> std::same_as<typename G::State>;

  { G::get_outcome(state, action) } -> std::same_as<std::optional<GameOutcome>>;
  { G::legal_actions(state) } -> std::same_as<torch::Tensor>;

  // IMPORTANT: The encoded state should always be from the perspective of
  // `state.player`.
  { G::encode_state(state) } -> std::same_as<torch::Tensor>;
};

}  // namespace concepts

export class Player {
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

inline constexpr auto Player::First = Player(true);
inline constexpr auto Player::Second = Player(false);

export class GameOutcome {
 public:
  const static GameOutcome Win;
  const static GameOutcome Loss;
  const static GameOutcome Draw;

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

inline constexpr auto GameOutcome::Win = GameOutcome(1);
inline constexpr auto GameOutcome::Loss = GameOutcome(-1);
inline constexpr auto GameOutcome::Draw = GameOutcome(0);

}  // namespace az
