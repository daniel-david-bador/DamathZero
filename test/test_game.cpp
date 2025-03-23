#include <gtest/gtest.h>

#include <print>

import damathzero;

using namespace DamathZero;

TEST(Game, InitialBoard) {
  auto expected_board = std::vector<int>(9, 0);
  EXPECT_EQ(Game::initial_board(), expected_board);
}

TEST(Game, ApplyAction) {
  auto expected_board = std::vector<int>(9, 0);
  expected_board[0] = 1;
  EXPECT_EQ(Game::apply_action(Game::initial_board(), 0, 1), expected_board);
}