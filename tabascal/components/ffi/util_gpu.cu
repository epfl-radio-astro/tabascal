#include "util_gpu.h"

#include <cuda_runtime.h>
#include <mutex>

namespace tabascal {
namespace gpu {

const cudaDeviceProp &get_device_prop() {
  static cudaDeviceProp prop;
  static std::once_flag flag;
  std::call_once(flag, [] { cudaGetDeviceProperties(&prop, 0); });
  return prop;
}

} // namespace gpu
} // namespace tabascal
