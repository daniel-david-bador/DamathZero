export module damathzero:node;

import std;

import :config;
import :network;
import :game;

namespace DamathZero {

export struct Node {
  using ID = int;

  Action action = -1;
  double prior = 0.0;
  Player player = -1;

  Node::ID parent_id = -1;
  std::vector<Node::ID> children = {};

  double value = 0.0;
  double visits = 0.0;
};

}  // namespace DamathZero
