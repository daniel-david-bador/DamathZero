export module damathzero:node;

import std;

import :config;
import :game;
import :network;

namespace DamathZero {

export struct Node {
  using ID = int;

  Board board;
  Player player;
  Action action;
  double prior;

  Node::ID parent = -1;
  std::vector<Node::ID> children = {};

  double value = 0.0;
  double visits = 0.0;

  constexpr Node(Board board, Player player=1, Action action=-1, double prior=0)
    : board(board), player(player), action(action), prior(prior) {}
};

}  // namespace DamathZero
