export module damathzero;

import std;

export import :game;
export import :mcts;
export import :network;

namespace DamathZero {

export class DamathZero {
  auto selfplay() -> void {
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
  }

 private:
  std::shared_ptr<Network> model_;
  std::shared_ptr<NodeStorage> nodes_;
};

}  // namespace DamathZero