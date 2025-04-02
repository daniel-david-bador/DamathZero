#include <gtest/gtest.h>

import std;
import alphazero;

using namespace AlphaZero;

TEST(MCTS, NodeRef) {
  auto nodes = NodeStorage{};

  auto first_id = nodes.create();
  auto node = nodes.as_ref(first_id);

  EXPECT_EQ(node->value, 0);
  EXPECT_EQ(node->visits, 0);

  node = nodes.create();

  EXPECT_NE(first_id, node.id);
  EXPECT_EQ(first_id, 0);
  EXPECT_EQ(node.id, 1);
}
