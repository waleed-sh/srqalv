#pragma once

#include "btkernel/types.hpp"

#include <array>
#include <unordered_map>

namespace btkernel {

bool triangle_ok(Spin a, Spin b, Spin c) noexcept;

class Wigner6jCache {
 public:

  double get(Spin j1, Spin j2, Spin j3, Spin l1, Spin l2, Spin l3);

  std::size_t size() const noexcept { return cache_.size(); }

 private:
  struct Key {
    std::array<Spin, 6> values{};

    bool operator==(const Key& other) const noexcept {
      return values == other.values;
    }
  };

  struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
      std::size_t seed = 0xcbf29ce484222325ull;
      for (const auto value : key.values) {
        seed ^= static_cast<std::size_t>(value + 0x9e3779b9);
        seed *= 0x100000001b3ull;
      }
      return seed;
    }
  };

  std::unordered_map<Key, double, KeyHash> cache_;

  static double compute(Spin j1, Spin j2, Spin j3, Spin l1, Spin l2, Spin l3);
};

}

