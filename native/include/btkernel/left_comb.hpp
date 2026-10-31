#pragma once

#include "btkernel/types.hpp"

namespace btkernel {

std::vector<Channel> enumerate_left_comb_channels(const std::vector<Spin>& two_js);

ChannelIndex build_channel_index(const std::vector<Channel>& channels);

}

