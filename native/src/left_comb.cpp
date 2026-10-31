#include "btkernel/left_comb.hpp"

#include "btkernel/wigner6j.hpp"

#include <functional>
#include <stdexcept>

namespace btkernel {

std::vector<Channel> enumerate_left_comb_channels(const std::vector<Spin>& two_js) {
  const int n = static_cast<int>(two_js.size());
  if (n < 2) {
    throw std::invalid_argument("A gauge-invariant vertex requires at least two legs.");
  }

  std::vector<Channel> out;
  if (n == 2) {
    if (two_js[0] == two_js[1]) {
      out.emplace_back();
    }
    return out;
  }
  if (n == 3) {
    if (triangle_ok(two_js[0], two_js[1], two_js[2])) {
      out.emplace_back();
    }
    return out;
  }

  Channel current(static_cast<std::size_t>(n - 3), 0);
  const int n_channels = static_cast<int>(current.size());

  std::function<void(int, Spin)> rec = [&](const int idx, const Spin previous) {
    if (idx == n_channels) {
      if (triangle_ok(previous, two_js[n - 2], two_js[n - 1])) {
        out.push_back(current);
      }
      return;
    }

    const Spin source = (idx == 0) ? two_js[0] : previous;
    const Spin partner = (idx == 0) ? two_js[1] : two_js[idx + 1];
    const Spin lo = std::abs(source - partner);
    const Spin hi = source + partner;
    for (Spin channel = lo; channel <= hi; ++channel) {
      if (!triangle_ok(source, partner, channel)) {
        continue;
      }
      current[static_cast<std::size_t>(idx)] = channel;
      rec(idx + 1, channel);
    }
  };

  rec(0, 0);
  return out;
}

ChannelIndex build_channel_index(const std::vector<Channel>& channels) {
  ChannelIndex index;
  index.reserve(channels.size() * 2 + 1);
  for (std::size_t i = 0; i < channels.size(); ++i) {
    index.emplace(channels[i], i);
  }
  return index;
}

}
