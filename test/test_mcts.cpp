#include <gtest/gtest.h>

import std;
import damathzero;

using namespace DamathZero;

TEST(MCTS, Search) {
  for (int i = 0; i < 100; i++) {
    auto nodes = std::make_shared<NodeStorage>();
    auto network = std::make_shared<Network>();
    auto mcts = MCTS{nodes};

    auto node = nodes->get_ref(nodes->create());

    auto [value, terminal] = Game::get_value_and_terminated(node->board, node->action);

    while (not terminal) {
      node = mcts.search(node.id, network);
      nodes->detach(node.id);

      std::tie(value, terminal) = Game::get_value_and_terminated(node->board, node->action);
    }

    EXPECT_TRUE(terminal);
    EXPECT_TRUE(value == 0.0 or value == 1.0);
  }
}