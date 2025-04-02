#include <gtest/gtest.h>

import std;
import alphazero;

using namespace AlphaZero;

TEST(Game, InitialBoard) {
  auto expected_board = std::vector<int>(9, 0);
  EXPECT_EQ(Game::initial_board(), expected_board);
}

TEST(Game, ApplyAction) {
  auto expected_board = std::vector<int>(9, 0);
  expected_board[0] = 1;
  EXPECT_EQ(Game::apply_action(Game::initial_board(), 0, 1), expected_board);
}