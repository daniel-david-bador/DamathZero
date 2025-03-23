module;

#include <memory>

export module damathzero;

export import :game;
export import :mcts;
export import :network;

namespace DamathZero {

export class DamathZero {
  auto selfplay() -> void {
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
  }

 private:
  std::shared_ptr<Network> model_;
  std::shared_ptr<NodeStorage> nodes_;
};

}  // namespace DamathZero