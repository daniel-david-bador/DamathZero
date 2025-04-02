#include <gtest/gtest.h>
#include <torch/torch.h>

import std;
import alphazero;

using namespace AlphaZero;

TEST(MCTS, Search) {
  for (int i = 0; i < 100; i++) {
    auto nodes = std::make_shared<NodeStorage>();
    auto network = std::make_shared<Network>();
    auto mcts = MCTS{nodes};

    auto node = nodes->as_ref(nodes->create());

    auto [value, terminal] =
        Game::get_value_and_terminated(node->board, node->action);

    while (not terminal) {
      node = mcts.search(node.id, network);
      nodes->detach(node.id);

      std::tie(value, terminal) =
          Game::get_value_and_terminated(node->board, node->action);
    }

    EXPECT_TRUE(terminal);
    EXPECT_TRUE(value == 0.0 or value == 1.0);
  }
}

TEST(MCTS, Expand) {
  auto nodes = std::make_shared<NodeStorage>();
  auto network = std::make_shared<Network>();
  auto mcts = MCTS{nodes};

  auto node = nodes->as_ref(nodes->create());
  EXPECT_TRUE(mcts.is_leaf(node.id));

  auto [_, policy] =
      network->forward(torch::tensor(node->board, torch::kFloat32));
  policy = torch::softmax(policy, -1);
  policy = policy.index({torch::tensor(Game::legal_actions(node->board))});
  policy /= policy.sum();

  mcts.expand(node.id, policy);

  EXPECT_FALSE(mcts.is_leaf(node.id));
}
