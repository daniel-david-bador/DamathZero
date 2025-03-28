module;

#include <assert.h>

export module damathzero:storage;

import std;

import :config;
import :game;
import :node;

namespace DamathZero {

export class NodeStorage {
  struct NodeRef {
    NodeStorage& storage;
    Node::ID id;

    constexpr NodeRef(NodeStorage& storage, Node::ID id)
        : storage(storage), id(id) {}

    constexpr auto operator=(Node::ID id) -> NodeRef& {
      this->id = id;
      return *this;
    }

    constexpr auto operator->() const -> const Node* {
      return &storage.get(id);
    }

    constexpr auto operator->() -> Node* { return &storage.get(id); }

    template <typename... Args>
    constexpr auto create_child(Args&&... args) -> void {
      auto child_id = storage.create(std::forward<Args>(args)...);
      storage.get(id).children.emplace_back(child_id);
      storage.get(child_id).parent_id = id;
    }
  };

 public:
  template <typename... Args>
  constexpr auto create(Args&&... args) -> Node::ID {
    nodes_.emplace_back(std::forward<Args>(args)...);
    return nodes_.size() - 1;
  }

  constexpr auto clear() -> void { nodes_.clear(); }

  constexpr auto as_ref(Node::ID id) -> NodeRef { return NodeRef(*this, id); }

  constexpr auto get(Node::ID id) -> Node& {
    assert(id >= 0 and id < static_cast<int>(nodes_.size()));
    return nodes_[id];
  }

  constexpr auto get(Node::ID id) const -> Node const& {
    assert(id >= 0 and id < static_cast<int>(nodes_.size()));
    return nodes_[id];
  }

 private:
  std::vector<Node> nodes_;
};

}  // namespace DamathZero
