#include <gtest/gtest.h>

import std;
import damathzero;

using namespace DamathZero;

TEST(MCTS, Search) {
  for (int i = 0; i < 100; i++) {
    auto nodes = std::make_shared<NodeStorage>();
    auto network = std::make_shared<Network>();
    auto mcts = MCTS{nodes};

    auto node_id = nodes->create();

    auto [value, terminal] = Game::get_value_and_terminated(
        nodes->get_board(node_id), nodes->get_action(node_id));

    while (not terminal) {
      node_id = mcts.search(node_id, network);
      nodes->detach(node_id);

      std::tie(value, terminal) = Game::get_value_and_terminated(
          nodes->get_board(node_id), nodes->get_action(node_id));
    }

    EXPECT_TRUE(terminal);
    EXPECT_TRUE(value == 0.0 or value == 1.0);
  }
}