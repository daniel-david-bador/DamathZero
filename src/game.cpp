module;

#include <torch/torch.h>

export module damathzero:game;

import std;

namespace DamathZero {

export using Action = int;

export using Feature = torch::Tensor;
export using Policy = torch::Tensor;
export using Value = torch::Tensor;

namespace Concepts {

class Player;

export template <typename G>
concept Game = requires(const G::State& state, Action action) {
  typename G::State;
  typename G::Network;

  std::is_base_of_v<torch::nn::Module, typename G::Network>;
  std::same_as<decltype(state.player), Player>;
  std::same_as<decltype(G::ActionSize), int>;

  { G::initial_state() } -> std::same_as<typename G::State>;

  { G::apply_action(state, action) } -> std::same_as<typename G::State>;

  { G::terminal_value(state, action) } -> std::same_as<std::optional<double>>;
  { G::legal_actions(state) } -> std::same_as<torch::Tensor>;

  // IMPORTANT: The encoded state should always be from the perspective of
  // `state.player`.
  { G::encode_state(state) } -> std::same_as<torch::Tensor>;
};

}  // namespace Concepts

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

}  // namespace DamathZero
