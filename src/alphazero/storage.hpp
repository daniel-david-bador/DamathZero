#pragma once

#include <assert.h>

#include "alphazero/node.hpp"

namespace az {

class NodeStorage {
  struct NodeRef {
    NodeStorage& storage;
    NodeId id;

    constexpr NodeRef(NodeStorage& storage, NodeId id)
        : storage(storage), id(id) {}

    constexpr auto operator=(NodeId id) -> NodeRef& {
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
      storage.get(id).add_child(child_id);
      storage.get(child_id).parent_id = id;
    }
  };

 public:
  template <typename... Args>
  constexpr auto create(Args&&... args) -> NodeId {
    nodes_.emplace_back(std::forward<Args>(args)...);
    return NodeId(nodes_.size() - 1);
  }

  constexpr auto clear() -> void { nodes_.clear(); }

  constexpr auto as_ref(NodeId id) -> NodeRef { return NodeRef(*this, id); }

  constexpr auto get(NodeId id) -> Node& {
    assert(id.is_valid());
    return nodes_[id.value()];
  }

  constexpr auto get(NodeId id) const -> Node const& {
    assert(id.is_valid());
    return nodes_[id.value()];
  }

 private:
  std::vector<Node> nodes_;
};

}  // namespace az
