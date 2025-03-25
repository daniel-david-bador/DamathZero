#include <gtest/gtest.h>
#include <torch/torch.h>

import std;
import damathzero;

TEST(DamathZero, SelfPlay) {
  auto model = std::make_shared<DamathZero::Network>();
  auto optimizer = std::make_shared<torch::optim::Adam>(
      model->parameters(), torch::optim::AdamOptions(0.001));

  auto random_device = std::random_device{};

  auto damath_zero = DamathZero::DamathZero{model, optimizer, random_device};
  damath_zero.learn();

  auto nodes = std::make_shared<DamathZero::NodeStorage>();
  auto mcts = DamathZero::MCTS{nodes};
  auto node = nodes->as_ref(nodes->create());

  auto [value, terminal] =
      DamathZero::Game::get_value_and_terminated(node->board, node->action);

  while (not terminal) {
    node = mcts.search(node.id, model);
    nodes->detach(node.id);

    std::println("hieieh");

    std::tie(value, terminal) =
        DamathZero::Game::get_value_and_terminated(node->board, node->action);
  }

  EXPECT_TRUE(terminal);
  EXPECT_TRUE(value == 0.0 or value == 1.0);
}
