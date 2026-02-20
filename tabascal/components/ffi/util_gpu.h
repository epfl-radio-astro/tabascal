#pragma once

#include <algorithm>
#include <cuda_runtime.h>

namespace tabascal {
namespace gpu {

const cudaDeviceProp &get_device_prop();

inline dim3 create_clamped_grid(int x, int y, int z) {
  const auto &prop = get_device_prop();
  return dim3(std::min<int>(x, prop.maxGridSize[0]),
              std::min<int>(y, prop.maxGridSize[1]),
              std::min<int>(z, prop.maxGridSize[2]));
}

} // namespace gpu
} // namespace tabascal
