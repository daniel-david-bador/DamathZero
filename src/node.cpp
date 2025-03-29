export module damathzero:node;

import std;

import :config;
import :game;

namespace DamathZero {

export struct Node {
  using ID = int;

  Player player = Player::First;
  Action action = -1;
  double prior = 0.0;

  Node::ID parent_id = -1;
  std::vector<Node::ID> children = {};

  double value = 0.0;
  double visits = 0.0;
};

}  // namespace DamathZero
