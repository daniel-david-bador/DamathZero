export module damathzero:node;

import std;

import :config;
import :game;
import :network;

namespace DamathZero {

export struct Node {
  using ID = std::size_t;

  Board board = Game::initial_board();
  Action action = -1;
  double prior = 0.0;

  Node::ID parent = -1;
  std::vector<Node::ID> children = {};

  double value = 0.0;
  double visits = 0.0;
};

}  // namespace DamathZero