module;

#include <assert.h>

export module alphazero:node;

import std;

import :config;
import :game;

namespace AlphaZero {

export struct NodeId {
 public:
  static const NodeId Invalid;

  constexpr auto is_valid() const -> bool { return value_ != -1; }
  constexpr auto value() const -> int {
    assert(value_ != -1);
    return value_;
  }
  constexpr explicit NodeId(int value) : value_(value) {}

  constexpr auto operator==(NodeId other) const -> bool {
    return other.value_ == value_;
  }

 private:
  int value_ = -1;
};

inline constexpr auto NodeId::Invalid = NodeId(-1);

export class Node {
 public:
  constexpr auto children() const {
    return std::views::iota(children_first.value(), children_last.value() + 1) |
           std::views::transform([](int value) { return NodeId(value); });
  }

  constexpr auto is_expanded() const -> bool {
    return children_first.is_valid();
  }

  constexpr auto add_child(NodeId child) -> void {
    assert(not children_first.is_valid() or
           children_last.value() + 1 == child.value());

    if (not children_first.is_valid()) {
      children_first = child;
    }

    children_last = child;
  }

  constexpr Node(Player player = Player::First, Action action = -1,
                 double prior = 0.0)
      : player(player), action(action), prior(prior) {}

 public:
  Player player = Player::First;
  Action action = -1;
  double prior = 0.0;

  NodeId parent_id = NodeId::Invalid;

  double value = 0.0;
  double visits = 0.0;

 private:
  NodeId children_first = NodeId::Invalid;
  NodeId children_last = NodeId::Invalid;
};

}  // namespace AlphaZero
