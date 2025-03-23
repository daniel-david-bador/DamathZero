module;

#include <torch/torch.h>

export module damathzero:mcts;

import std;

import :game;
import :network;
import :config;

namespace DamathZero {

export struct Node {
  using ID = int;

  Board board = Game::initial_board();
  Action action = -1;
  double prior = 0.0;

  Node::ID parent = -1;
  std::vector<Node::ID> children = {};

  double value = 0.0;
  double visits = 0.0;
};


export class NodeStorage {
  struct NodeRef {
    NodeStorage& storage;
    Node::ID id;

    constexpr NodeRef(NodeStorage& storage, Node::ID id) : storage(storage), id(id) {}

    constexpr auto operator=(Node::ID id) -> NodeRef& {
      this->id = id;
      return *this;
    }

    constexpr auto operator->() const -> const Node* {
      return &storage.get(id);
    }

    constexpr auto operator->() -> Node* {
      return &storage.get(id);
    }
  };

 public:
  template <typename... Args>
  constexpr auto create(Args&&... args) -> Node::ID {
    nodes_.emplace_back(std::forward<Args>(args)...);
    return nodes_.size() - 1;
  }

  template <typename... Args>
  constexpr auto create_child(Node::ID id, Args&&... args) -> void {
    auto child_id = create(std::forward<Args>(args)...);
    nodes_[id].children.emplace_back(child_id);
  }

  constexpr auto get_ref(Node::ID id) -> NodeRef {
    return NodeRef(*this, id);
  }

  constexpr auto get(Node::ID id) -> Node& {
    assert(id >= 0 and static_cast<size_t>(id) < nodes_.size());
    return nodes_[id];
  }

  constexpr auto get(Node::ID id) const -> Node const& {
    assert(id >= 0 and static_cast<size_t>(id) < nodes_.size());
    return nodes_[id];
  }


  constexpr auto detach(Node::ID id) -> void {
    assert(id >= 0 and static_cast<size_t>(id) < nodes_.size());
    nodes_[id].parent = -1;
  }

 private:
  std::vector<Node> nodes_;
};

export class MCTS {
 public:
  MCTS(std::shared_ptr<NodeStorage> nodes) : nodes_(nodes) {};

  template <typename Network>
  constexpr auto search(Node::ID root_id, std::shared_ptr<Network> network)
      -> Node::ID {
    torch::NoGradGuard no_grad;
    for (auto _ : std::views::iota(0, Config::num_simulations)) {
      auto node  = nodes_->get_ref(root_id);

      while (not is_leaf(node.id))
        node = highest_child_score(node.id);

      auto [value, terminal] = Game::get_value_and_terminated(node->board, node->action);
      value = Game::get_opponent_value(value);

      if (not terminal) {
        auto [value_tensor, policy] = network->forward(
            torch::tensor(node->board, torch::kFloat32));
        policy = torch::softmax(policy, -1);
        policy = policy.index({torch::tensor(Game::legal_actions(node->board))});
        policy /= policy.sum();

        value = value_tensor.template item<double>();
        expand(node.id, policy);
      }

      backpropagate(node.id, value);
    }

    return highest_child_visits(root_id);
  }

 private:
  constexpr auto is_leaf(Node::ID id) const -> bool {
    auto& node = nodes_->get(id);
    return node.children.empty();
  };

  constexpr auto score(Node::ID id) const -> double {
    auto& node = nodes_->get(id);
    auto& parent = nodes_->get(node.parent);

    auto mean =
        node.visits > 0.0 ? 1 - ((node.value / node.visits) + 1) / 2.0 : 0.0;
    return mean + node.prior * 2 * std::sqrt(parent.visits) / (1 + node.visits);
  };

  constexpr auto highest_child_score(Node::ID id) -> Node::ID {
    auto& node = nodes_->get(id);
    assert(not node.children.empty());

    auto highest_score = [this](Node::ID a, Node::ID b) {
      return score(a) < score(b);
    };

    return *std::ranges::max_element(node.children, highest_score);
  };

  constexpr auto highest_child_visits(Node::ID id) -> Node::ID {
    auto& node = nodes_->get(id);
    assert(not node.children.empty());

    auto highest_visits = [this](Node::ID a, Node::ID b) {
      return nodes_->get(a).visits < nodes_->get(b).visits;
    };

    return *std::ranges::max_element(node.children, highest_visits);
  };

  constexpr auto expand(Node::ID id, const Policy& policy) -> void {
    auto node = nodes_->get_ref(id);
    auto legal_actions = Game::legal_actions(node->board);
    for (size_t i = 0; i < legal_actions.size(); i++) {
      auto action = legal_actions[i];
      auto prior = policy[i].item<double>();
      auto child_board = Game::apply_action(node->board, action, 1);
      child_board = Game::change_perpective(child_board, -1);
      nodes_->create_child(id, child_board, action, prior, id);
    }
  };

  constexpr auto backpropagate(Node::ID id, double value) -> void {
    auto& node = nodes_->get(id);

    node.visits += 1;
    node.value += value;

    value = Game::get_opponent_value(value);
    if (node.parent > 0)
      backpropagate(node.parent, value);
  };

  std::shared_ptr<NodeStorage> nodes_;
};
}  // namespace DamathZero