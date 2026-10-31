#pragma once

#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace btkernel {

// all spins use doubled labels
using Spin = int;

// stores a_2 through a_{N-2}, with the final singlet coupling implicit
using Channel = std::vector<Spin>;

using ComplexVector = std::vector<std::complex<double>>;

struct TripleSign {
  int i = 0;
  int j = 0;
  int k = 0;
  int sign = 0;
};

struct ChannelHash {
  std::size_t operator()(const Channel& channel) const noexcept {
    std::size_t seed = 1469598103934665603ull;
    for (const auto value : channel) {
      seed ^= static_cast<std::size_t>(value + 0x9e3779b9);
      seed *= 1099511628211ull;
    }
    return seed;
  }
};

struct ChannelEqual {
  bool operator()(const Channel& a, const Channel& b) const noexcept {
    return a == b;
  }
};

using ChannelIndex = std::unordered_map<Channel, std::size_t, ChannelHash, ChannelEqual>;

}

