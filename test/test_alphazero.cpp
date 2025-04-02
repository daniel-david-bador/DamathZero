#include <gtest/gtest.h>
#include <torch/torch.h>

import std;
import alphazero;

TEST(AlphaZero, SelfPlay) {
  auto model = std::make_shared<AlphaZero::Network>();
  auto optimizer = std::make_shared<torch::optim::Adam>(
      model->parameters(), torch::optim::AdamOptions(0.001));

  auto random_device = std::random_device{};

  auto damath_zero = AlphaZero::AlphaZero{model, optimizer, random_device};
  damath_zero.learn();

  auto nodes = std::make_shared<AlphaZero::NodeStorage>();
  auto mcts = AlphaZero::MCTS{nodes};
  auto node = nodes->as_ref(nodes->create());

  auto [value, terminal] =
      AlphaZero::Game::get_value_and_terminated(node->board, node->action);

  while (not terminal) {
    node = mcts.search(node.id, model);
    nodes->detach(node.id);

    std::tie(value, terminal) =
        AlphaZero::Game::get_value_and_terminated(node->board, node->action);
  }

  EXPECT_TRUE(terminal);
  EXPECT_TRUE(value == 0.0 or value == 1.0);
}
