#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <mutex>

#include <ATen/native/cuda/arc_flag.h>
#include <ATen/Context.h>

// [JS] P2P define
#include <queue>
#include <ATen/cuda/CUDAEvent.h>

// Half precision
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <c10/cuda/CUDACachingAllocator.h>

#define find(n) (32 * (unsigned int)(n / 1024) + (n % 32))
#define mask(n) (0x80000000 >> (unsigned int)((n % 1024) / 32))

// [JS] Arcp2p Setting flag define
#define ARC_FLAG_VDNN  (1U << 0)
#define ARC_FLAG_FP16  (1U << 1)
#define ARC_FLAG_CSR   (1U << 2)
#define ARC_FLAG_SSD   (1U << 3)
#define ARC_FLAG_TESLA (1U << 4)
#define ARC_FLAG_RAID0 (1U << 5)
#define ARC_FLAG_DEBUG (1U << 6)
// [JS] 7~11 bit will be used for arc_vm (device) cudamalloc size
#define ARC_MEMSIZE_MASK  (0x00000F80)
// 12~16 bit will be used for arc_vm (p2p) cudamalloc size
#define ARC_P2PSIZE_MASK  (0x0001F000)
#define ARC_FLAG_TIMER (1U << 17)
#define ARC_MEMSIZE_SHIFT (7)

using namespace at::cuda;
__global__ void double_scale(__half *din, double *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize)  dout[idx] = (double)__half2float(din[idx]);
}

__global__ void float_scale(__half *din, float *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize)  dout[idx] = __half2float(din[idx]);
}

__global__ void zero_insert_double(unsigned int *bit, unsigned int *nz_pos, float* din, double *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize) {
    int count = -1;
    if ((unsigned int)(bit[find(idx)] & mask(idx)) > 0) {
      for (int i = (int)(idx / 32) * 32; i < idx + 1; i++) {
        unsigned int mask = bit[find(i)] & mask(i);
        if (mask > 0)  count += 1;
      }
    }

    if (count == -1)  dout[idx] = 0.0;
    else {
      if ((unsigned int)(idx / 32) == 0) {
        dout[idx] = (double)din[count + 0];
      } else {
        dout[idx] = (double)din[count + nz_pos[(unsigned int)(idx / 32) - 1]];
      }
    }
  }
}

__global__ void zero_insert_float(unsigned int *bit, unsigned int *nz_pos, float* din, float *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize) {
    int count = -1;
    if ((unsigned int)(bit[find(idx)] & mask(idx)) > 0) {
      for (int i = (int)(idx / 32) * 32; i < idx + 1; i++) {
        unsigned int mask = bit[find(i)] & mask(i);
        if (mask > 0)  count += 1;
      }
    }

    if (count == -1)  dout[idx] = 0.0f;
    else {
      if ((unsigned int)(idx / 32) == 0) {
        dout[idx] = din[count + 0];
      } else {
        dout[idx] = din[count + nz_pos[(unsigned int)(idx / 32) - 1]];
      }
    }
  }
}

namespace at { namespace native {

using namespace at::cuda;

ARC_memory arc_vm;

typedef struct {
  uint64_t addr;
  uint64_t size;
  uint64_t offs;
  arcp2p_dir dir;
  arcp2p_cpl *p_cpl;

  c10::Storage *stor;
  arcp2p_info *info;
  cudaStream_t str;

  // Additional information for post task
  // for GPU to SSD case,
  //  - We need to keep GPU memory until transfer is done.
  //  - Release right after process is completed
  //  - Therefore, we need to keep "c10::Storage" class at here
  // for SSD to GPU case,
  //  - We need to keep required data for half-to-float conversion
  //  - These are only for FP16 and CSR case
} req_element;

std::queue<req_element> req_queue;

ARC_memory::ARC_memory(): global_tensor_id_(0), cur_back_num(0), hard_training(false), relu_thru(false), mapping(false),
    gradient_map_accum(0), weight_accum(0), misc_accum(0), isTimer(false),
    isVDNN(false), isFP16(false), isCSR(false), isUsingSSD(false), isTesla(false), isDebug(false),
    device_sz(0), max_device(0), p2p_sz(0), max_p2p(0) {

  on_the_fly = 0;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&endEvent);

  /*
  liveness_result = new bool[3][NUM_TENSOR];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < NUM_TENSOR; j++) {
      liveness_result[i][j] = false;
    }
  }
  */
  int greatestPriority, leastPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  cudaStreamCreateWithPriority(&arc_stream, cudaStreamNonBlocking, leastPriority);

  fp16_ptr_arr = new uint64_t[NUM_TENSOR];
  bit_ptr_arr = new uint64_t[NUM_TENSOR];
  pos_ptr_arr = new uint64_t[NUM_TENSOR];
  resize_arr = new int[NUM_TENSOR];
  numel_arr = new size_t[NUM_TENSOR];
  elem_arr = new int[NUM_TENSOR];
  cpl_flu_ptr_arr = new uint64_t[NUM_TENSOR];
  cpl_pre_ptr_arr = new uint64_t[NUM_TENSOR];
  offset_arr = new uint64_t[NUM_TENSOR];
  dir_arr = new arcp2p_dir[NUM_TENSOR];

  event_arr_d2h = new bool[NUM_TENSOR];
  event_arr_h2d = new bool[NUM_TENSOR];

  for(int i = 0; i < NUM_TENSOR; i++) {
    event_arr_d2h[i] = false;
    event_arr_h2d[i] = false;
    feature_map_accum[i] = 0;
  }

  memset(dir_arr, 0, sizeof(arcp2p_dir) * NUM_TENSOR);
}

ARC_memory::~ARC_memory() {
  if (device_sz > 0 && isVDNN) {
    cudaFree(deviceAddr);
    delete[] deviceTable;
    delete[] device_page_map;
    delete[] device_page_map_rev;
  }

  if (p2p_sz > 0 && isVDNN) {
    cudaFree(p2pAddr);
    delete[] p2pTable;
    delete[] p2p_page_map;
  }

  delete[] fp16_ptr_arr;
  delete[] bit_ptr_arr;
  delete[] pos_ptr_arr;
  delete[] resize_arr;
  delete[] numel_arr;
  delete[] elem_arr;
  delete[] cpl_flu_ptr_arr;
  delete[] cpl_pre_ptr_arr;
  delete[] offset_arr;
  delete[] dir_arr;

  delete[] event_arr_d2h;
  delete[] event_arr_h2d;

  if (true == isUsingSSD)
  {
    arcp2p_synchronize(arc_handle);
    if (true == isTesla)
    {
      arcp2p_bar_detach(arc_handle);
    }
    arcp2p_release(arc_handle);
  }
}

void ARC_memory::device_malloc(void** gpu_ptr, size_t size) {
  int reqBlk = std::ceil((double)size / (double)BLK_SZ);
  int blkCheck = 0;
  int retryCnt = 0;

  if (device_sz == 0) return;

  if (reqBlk == 0) return;

  devStartBlk = &devBlk_0_4;
  devMaxBlk = max_device;

  while (true) {
    for (int i = *devStartBlk; i < devMaxBlk; i++) {
      blkCheck += 1;
      if (deviceTable[i] != 0) {
        if (deviceTable[i] == 2) {
          std::cout << "Table meet with reverse" << std::endl;
        }
        if (device_page_map[i] == 0) {
          std::cout << "device_page_map[" << i << "] is zero, size: " << size << ", " << blkCheck << ", " << reqBlk << std::endl;
          exit(1);
        }
        i += device_page_map[i] - 1;
        blkCheck = 0; *devStartBlk = i + 1;
        continue;
      }

      if (blkCheck == reqBlk) {
        device_page_map[*devStartBlk] = reqBlk;
        *gpu_ptr = (void* )((size_t)deviceAddr + (*devStartBlk * BLK_SZ));

        for (int i = *devStartBlk; i < *devStartBlk + reqBlk; i++) {
          deviceTable[i] = 1;
        }

        *devStartBlk += reqBlk;
        dev_freeBlk -= reqBlk;

        return;
      }
    }
    blkCheck = 0;
    *devStartBlk = 0;

    if (retryCnt++ > 2) {
      if (isDebug) {
        std::cout << "dev malloc failed: " << (double)size / 1024 / 1024 << ", " << device_occupancy() << std::endl;
      }
      *gpu_ptr = NULL;
      return;
    }
  }
}

void ARC_memory::device_malloc_reverse(void** gpu_ptr, size_t size) {
  int reqBlk = std::ceil((double)size / (double)BLK_SZ);
  int blkCheck = 0;
  int retryCnt = 0;

  if (device_sz == 0) return;

  if (reqBlk == 0) return;

  devStartBlk_rev = &devBlk_128_rev;
  devMaxBlk = 0;

  while (true) {
    for (int i = *devStartBlk_rev; i >= devMaxBlk; i--) {
      blkCheck += 1;
      if (deviceTable[i] != 0) {
        if (deviceTable[i] == 1) {
          std::cout << "Table meet at first table" << std::endl;
        }
        if (device_page_map_rev[i] == 0) {
          std::cout << "device_page_map_rev[" << i << "] is zero, size: " << size << ", " << blkCheck << ", " << reqBlk << ", " << deviceTable[i] << std::endl;
          exit(1);
        }
        i -= (device_page_map_rev[i] - 1);
        blkCheck = 0; *devStartBlk_rev = i - 1;
        continue;
      }

      if (blkCheck == reqBlk) {
        device_page_map_rev[*devStartBlk_rev] = reqBlk;
        *gpu_ptr = (void* )((size_t)deviceAddr + ((*devStartBlk_rev - reqBlk + 1) * BLK_SZ));

        for (int i = *devStartBlk_rev; i >= *devStartBlk_rev - reqBlk + 1; i--) {
          deviceTable[i] = 2;
        }

        *devStartBlk_rev -= reqBlk;
        dev_freeBlk -= reqBlk;

        return;
      }
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
    blkCheck = 0;
    *devStartBlk_rev = max_device - 1;

    if (retryCnt++ > 2) {
      std::cout << "dev malloc reverse failed: " << (double)size / 1024 / 1024 << ", " << device_occupancy() << std::endl;
      *gpu_ptr = NULL;
      return;
    }
  }
}

void ARC_memory::device_free(void* addr, size_t size) {
  int startBlk = ((size_t)addr - (size_t)deviceAddr) / BLK_SZ;
  int reqBlk = std::ceil((double)size / (double)BLK_SZ);

  if (device_sz == 0) return;

  device_page_map[startBlk] = 0;

  bool reverse = false;
  if (deviceTable[startBlk] == 2) {
    reverse = true;
  }

  if (isDebug)
    std::cout << "device_free addr: " << addr << ", size: " << size << ", reverse: " << reverse << std::endl;

  for (unsigned int i = startBlk; i < startBlk + reqBlk; i++) {
    deviceTable[i] = 0;
  }

  dev_freeBlk += reqBlk;

  if (!reverse) {
    devBlk_0_4 = std::min(devBlk_0_4, startBlk);
  } else {
    devBlk_128_rev = std::max(devBlk_128_rev, startBlk + reqBlk - 1);
  }
}

size_t ARC_memory::device_occupancy_size() {
  return dev_freeBlk * BLK_SZ;
}

size_t ARC_memory::p2p_occupancy_size() {
  return p2p_freeBlk * BLK_SZ;
}

double ARC_memory::device_occupancy() {
  return dev_freeBlk / (double)max_device;
}

double ARC_memory::device_occupancy_future(size_t size) {
  unsigned int reqBlk = std::ceil((double)size / (double)BLK_SZ);

  return (dev_freeBlk - reqBlk) / (double)max_device;
}


double ARC_memory::p2p_occupancy() {
  return p2p_freeBlk / (double)max_p2p;
}

void ARC_memory::p2p_malloc(void** gpu_ptr, size_t size) {
  int reqBlk = std::ceil((double)size / (double)BLK_SZ);
  int blkCheck = 0;
  int retryCnt = 0;

  if (p2p_sz == 0) return;

  if (reqBlk == 0) return;

  if (isDebug) {
    std::cout << "p2p malloc size test sampling: " << (double)size / 1024 / 1024 << std::endl;
  }

/*
  if (size < ((size_t)1 << 22)) {
    p2pStartBlk = &p2pBlk_0_4;
    p2pMaxBlk = pinit_4m;
  } else if (size < ((size_t)1 << 23)) {
    p2pStartBlk = &p2pBlk_4_8;
    p2pMaxBlk = pinit_8m;
  } else if (size < ((size_t)1 << 24)) {
    p2pStartBlk = &p2pBlk_8_16;
    p2pMaxBlk = pinit_16m;
  } else if (size < ((size_t)1 << 25)) {
    p2pStartBlk = &p2pBlk_16_32;
    p2pMaxBlk = pinit_32m;
  } else if (size < ((size_t)1 << 26)) {
    p2pStartBlk = &p2pBlk_32_64;
    p2pMaxBlk = pinit_64m;
  } else {
    p2pStartBlk = &p2pBlk_64;
    p2pMaxBlk = max_p2p;
  }
*/
  p2pStartBlk = &p2pBlk_0_4;
  p2pMaxBlk = max_p2p;

  while (true) {
    for (int i = *p2pStartBlk; i < p2pMaxBlk; i++) {
      blkCheck += 1;
      if (p2pTable[i]) {
        if (p2p_page_map[i] == 0) {
          std::cout << "p2p_page_map[" << i << "] is zero, size: " << size << ", " << blkCheck << ", " << reqBlk << std::endl;
          exit(1);
        }
        i += p2p_page_map[i] - 1;
        blkCheck = 0; *p2pStartBlk = i + 1;
        continue;
      }

      if (blkCheck == reqBlk) {
        p2p_page_map[*p2pStartBlk] = reqBlk;
        *gpu_ptr = (void* )((size_t)p2pAddr + (*p2pStartBlk * BLK_SZ));

        for (int i = *p2pStartBlk; i < *p2pStartBlk + reqBlk; i++) {
          p2pTable[i] = true;
        }

        *p2pStartBlk += reqBlk;
        p2p_freeBlk -= reqBlk;

        return;
      }
    }
    blkCheck = 0;
/*
    if (size < ((size_t)1 << 22)) *p2pStartBlk = 0;
    else if (size < ((size_t)1 << 23)) *p2pStartBlk = pinit_4m;
    else if (size < ((size_t)1 << 24)) *p2pStartBlk = pinit_8m;
    else if (size < ((size_t)1 << 25)) *p2pStartBlk = pinit_16m;
    else if (size < ((size_t)1 << 26)) *p2pStartBlk = pinit_32m;
    else *p2pStartBlk = pinit_64m;
*/
    *p2pStartBlk = 0;

    if (retryCnt++ > 2) {
      std::cout << "p2p malloc failed: " << (double)size / 1024 / 1024 << ", " << p2p_occupancy() << std::endl;
      *gpu_ptr = NULL;
      return;
    }
  }
}

void ARC_memory::p2p_free(void* addr, size_t size) {
  int startBlk = ((size_t)addr - (size_t)p2pAddr) / BLK_SZ;
//  int reqBlk = std::ceil((double)size / (double)BLK_SZ);
  int reqBlk = p2p_page_map[startBlk];

  if (p2p_sz == 0) return;

  p2p_page_map[startBlk] = 0;

  for (unsigned int i = startBlk; i < startBlk + reqBlk; i++) {
    p2pTable[i] = false;
  }

  p2p_freeBlk += reqBlk;

/*
  if (size < ((size_t)1 << 22)) p2pBlk_0_4 = std::min(p2pBlk_0_4, startBlk);
  else if (size < ((size_t)1 << 23)) p2pBlk_4_8 = std::min(p2pBlk_4_8, startBlk);
  else if (size < ((size_t)1 << 24)) p2pBlk_8_16 = std::min(p2pBlk_8_16, startBlk);
  else if (size < ((size_t)1 << 25)) p2pBlk_16_32 = std::min(p2pBlk_16_32, startBlk);
  else if (size < ((size_t)1 << 26)) p2pBlk_32_64 = std::min(p2pBlk_32_64, startBlk);
  else p2pBlk_64 = std::min(p2pBlk_64, startBlk);
*/
  p2pBlk_0_4 = std::min(p2pBlk_0_4, startBlk);
}

void* ARC_memory::get_fp16_addr(int tid) {
  return (void *)fp16_ptr_arr[tid];
}

void ARC_memory::set_fp16_addr(int tid, uint64_t addr) {
  fp16_ptr_arr[tid] = addr;
}

void* ARC_memory::get_device_addr() {
  return p2pAddr;
}

uint64_t ARC_memory::get_device_sz() {
  return p2p_sz;
}

void* ARC_memory::get_bit_addr(int tid) {
  return (void *)bit_ptr_arr[tid];
}

void ARC_memory::set_bit_addr(int tid, uint64_t addr) {
  bit_ptr_arr[tid] = addr;
}

void* ARC_memory::get_pos_addr(int tid) {
  return (void *)pos_ptr_arr[tid];
}

void ARC_memory::set_pos_addr(int tid, uint64_t addr) {
  pos_ptr_arr[tid] = addr;
}

int ARC_memory::get_resize(int tid) {
  return resize_arr[tid];
}

void ARC_memory::set_resize(int tid, int resize) {
  resize_arr[tid] = resize;
}

size_t ARC_memory::get_numel(int tid) {
  return numel_arr[tid];
}

void ARC_memory::set_numel(int tid, size_t numel) {
  numel_arr[tid] = numel;
}

int ARC_memory::get_elem(int tid) {
  return elem_arr[tid];
}

void ARC_memory::set_elem(int tid, int elem) {
  elem_arr[tid] = elem;
}

void* ARC_memory::get_cpl_addr(int tid, arcp2p_dir dir) {
  if (arcp2p_gputossd == dir) {
    return (void *)cpl_flu_ptr_arr[tid];
  } else if (arcp2p_ssdtogpu == dir) {
    return (void *)cpl_pre_ptr_arr[tid];
  } else {
    return nullptr;
  }
}

void ARC_memory::set_cpl_addr(int tid, arcp2p_dir dir, void *addr) {
  if (arcp2p_gputossd == dir) {
    cpl_flu_ptr_arr[tid] = (uint64_t)addr;
  } else if (arcp2p_ssdtogpu == dir) {
    cpl_pre_ptr_arr[tid] = (uint64_t)addr;
  }
}

uint64_t* ARC_memory::get_offset_ptr(int tid) {
  return &offset_arr[tid];
}

arcp2p_dir ARC_memory::get_dir(int tid) {
  return dir_arr[tid];
}

void ARC_memory::set_dir(int tid, arcp2p_dir dir) {
  dir_arr[tid] = dir;
}

bool ARC_memory::is_timer(void) {
  return isTimer;
}

bool ARC_memory::is_vdnn(void) {
  return isVDNN;
}

bool ARC_memory::is_fp16(void) {
  return isFP16;
}

bool ARC_memory::is_csr(void) {
  return isCSR;
}

bool ARC_memory::is_using_ssd(void) {
  return isUsingSSD;
}

bool ARC_memory::is_debug(void) {
  return isDebug;
}

void ARC_memory::Arcp2pSetting(int flags) {
  printf("Arcp2pSetting : 0x%x\n", flags);

  uint64_t device_in_gb;
  device_in_gb = (flags & ARC_MEMSIZE_MASK) >> ARC_MEMSIZE_SHIFT;
  device_sz = device_in_gb << 30;
  max_device = device_sz / BLK_SZ;

  uint64_t p2p_in_gb;
  p2p_in_gb = (flags & ARC_P2PSIZE_MASK) >> 12;
  p2p_sz = p2p_in_gb << 30;
  max_p2p = p2p_sz / BLK_SZ;

  init_4m = ((size_t)(max_device * 0.1));
  init_16m = ((size_t)(max_device * 0.3));
  init_64m = ((size_t)(max_device * 0.7));
  init_128m = ((size_t)(max_device * 0.95));

  devBlk_0_4 = 0;
  devBlk_4_16 = init_4m;
  devBlk_16_64 = init_16m;
  devBlk_64_128 = init_64m;
  devBlk_128 = init_128m;

  devBlk_0_4_rev = init_4m - 1;
  devBlk_4_16_rev = init_16m - 1;
  devBlk_16_64_rev = init_64m - 1;
  devBlk_64_128_rev = init_128m - 1;
  devBlk_128_rev = max_device - 1;

  // CycleGAN batch-16
/*
  pinit_4m = ((size_t)(max_p2p * 0.1));
  pinit_8m = ((size_t)(max_p2p * 0.15));
  pinit_16m = ((size_t)(max_p2p * 0.2));
  pinit_32m = ((size_t)(max_p2p * 0.3));
  pinit_64m = ((size_t)(max_p2p * 0.7));
*/

  // BERT batch-14
/*
  pinit_4m = ((size_t)(max_p2p * 0.15));
  pinit_8m = ((size_t)(max_p2p * 0.3));
  pinit_16m = ((size_t)(max_p2p * 0.45));
  pinit_32m = ((size_t)(max_p2p * 0.55));
  pinit_64m = ((size_t)(max_p2p * 0.9));
*/

  p2pBlk_0_4 = 0;
/*
  p2pBlk_4_8 = pinit_4m;
  p2pBlk_8_16 = pinit_8m;
  p2pBlk_16_32 = pinit_16m;
  p2pBlk_32_64 = pinit_32m;
  p2pBlk_64 = pinit_64m;
*/

  printf("Device memory size = %ld GB\n", device_in_gb);
  printf("P2P memory size = %ld GB\n", p2p_in_gb);

  if (device_in_gb > 0) {
    cudaMalloc(&deviceAddr, device_sz);
    deviceTable = new short[max_device];
    memset(deviceTable, 0, sizeof(short) * max_device);

    device_page_map = new unsigned int[max_device];
    for (int i = 0; i < max_device; i++) {
      device_page_map[i] = 0;
    }

    device_page_map_rev = new unsigned int[max_device];
    for (int i = 0; i < max_device; i++) {
      device_page_map_rev[i] = 0;
    }

    dev_freeBlk = (double)max_device;
  }

  if (p2p_in_gb > 0) {
    cudaMalloc(&p2pAddr, p2p_sz);
    p2pTable = new bool[max_p2p];
    memset(p2pTable, 0, sizeof(bool) * max_p2p);

    p2p_page_map = new unsigned int[max_p2p];
    for (int i = 0; i < max_p2p; i++) {
      p2p_page_map[i] = 0;
    }

    p2p_freeBlk = (double)max_p2p;
  }

  if (flags & ARC_FLAG_TIMER) {
    printf("Timer profiler set\n");
    isTimer = true;
  }

  if (flags & ARC_FLAG_VDNN) {
    printf("vDNN flag set\n");
    isVDNN = true;
  }

  if (flags & ARC_FLAG_FP16) {
    printf("FP16 flag set\n");
    isVDNN = true;
    isFP16 = true;
  }

  if (flags & ARC_FLAG_CSR) {
    printf("CSR flag set\n");
    isVDNN = true;
    isFP16 = true;
    isCSR = true;
  }

  if (flags & ARC_FLAG_TESLA) {
    printf("Tesla flag set\n");
    isTesla = true;
  }

  if (flags & ARC_FLAG_SSD) {
    printf("SSD flag set\n");
    // [JS] P2P
    isVDNN = true;
    isUsingSSD = true;
    last_allocated_offset = 0;

    const char *nvme_path_tesla[PATH_LENGTH] = {"0000:65:00.00", "0000:66:00.00"}; // TESLA
    const char *nvme_path_quadro[PATH_LENGTH] = {"0000:85:00.00", ""}; // QUADRO
    const int nvme_cnt = (flags & ARC_FLAG_RAID0)?2:1;

    printf("RAID0 flag check, device cnt %d\n", nvme_cnt);

    void* lib_handle;
    if (!(lib_handle = dlopen("/usr/local/lib/libarcp2p.so", RTLD_LAZY))) {
      fprintf(stderr, "%s\n", dlerror());
      return;
    }

    arcp2p_initialize = (arcp2p_type1_fn)dlsym(lib_handle, "ARCP2P_initialize");
    if(dlerror()) { fprintf(stderr, "Error linking\n"); return; }

    arcp2p_release    = (arcp2p_type2_fn)dlsym(lib_handle, "ARCP2P_release");
    if(dlerror()) { fprintf(stderr, "Error linking\n"); return; }

    arcp2p_bar_attach = (arcp2p_type3_fn)dlsym(lib_handle, "ARCP2P_bar_attach");
    if(dlerror()) { fprintf(stderr, "Error linking\n"); return; }

    arcp2p_bar_detach = (arcp2p_type2_fn)dlsym(lib_handle, "ARCP2P_bar_detach");
    if(dlerror()) { fprintf(stderr, "Error linking\n"); return; }

    arcp2p_transfer   = (arcp2p_type4_fn)dlsym(lib_handle, "ARCP2P_transfer");
    if(dlerror()) { fprintf(stderr, "Error linking\n"); return; }

    arcp2p_completion = (arcp2p_type2_fn)dlsym(lib_handle, "ARCP2P_completion");
    if(dlerror()) { fprintf(stderr, "Error linking\n"); return; }

    arcp2p_synchronize = (arcp2p_type2_fn)dlsym(lib_handle, "ARCP2P_synchronize");
    if(dlerror()) { fprintf(stderr, "Error linking\n"); return; }

    if (true == isTesla) {
      arc_handle = arcp2p_initialize(nvme_path_tesla, nvme_cnt);
    } else {
      arc_handle = arcp2p_initialize(nvme_path_quadro, nvme_cnt);
    }
  } else { // if not ssd
    isUsingSSD = false;
  }

  if (flags & ARC_FLAG_DEBUG) {
    printf("Debug mode on\n");
    isDebug = true;
    at::globalContext().ARCGlobal.turnOnDebugMode();
  } else {
    isDebug = false;
  }
}

// bar attach
int  ARC_memory::Arcp2pBarMapping(uint64_t addr, uint64_t size) {
  return arcp2p_bar_attach(arc_handle, addr, size);
}

// submission
void ARC_memory::Arcp2pSubmission(uint64_t addr, uint64_t size, uint64_t *p_offs,
    arcp2p_cpl *p_cpl, arcp2p_dir dir, c10::Storage *stor, arcp2p_info *info, cudaStream_t str) {
  uint64_t offset, aligned_size;

  const uint64_t prp_align_size = (1UL << 12);
  const uint64_t prp_align_mask = (prp_align_size - 1);

  // align up the size value
  if (size & prp_align_mask) {
    aligned_size = (size + prp_align_size - 1) & (~prp_align_mask);
  } else {
    aligned_size = size;
  }

  if (arcp2p_gputossd == dir) {
    // flush case, need to allocate nvme area
    offset = last_allocated_offset;
    last_allocated_offset = last_allocated_offset + aligned_size;

    *p_offs = offset;
  } else {
    // prefetch case, handle requested nvme offset
    offset = *p_offs;
  }

  req_element req;
  req.addr = addr;
  req.size = aligned_size;
  req.dir = dir;
  req.stor = stor;
  req.info = info;
  req.str = str;

  req.offs = offset;
  req.p_cpl = p_cpl;

  req.p_cpl->requested = true;
  req.p_cpl->arc_handle = arc_handle;

  if (true == isTesla) {
    // directly deliver transfer request to arcp2p library, only for tesla
    arcp2p_transfer(arc_handle, addr, offset, aligned_size, req.p_cpl, dir);
  } else {
    // for quadro, we need to attach bar range before transfer
    // check that queue is empty, else case will be handled at completion function
    if (req_queue.empty()) {
      printf("Transfer directly\n");
      //arcp2p_bar_attach(arc_handle, addr, size);
      // debug code. retry 10 times
      int retrycnt = 0;
      while(ARCP2P_NO_ERROR != arcp2p_bar_attach(arc_handle, addr, size)) {
        retrycnt ++;
        printf("Bar attach failed, retry %d/10\n", retrycnt);
        if (retrycnt >= 10) {
          break;
        }
        arcp2p_bar_detach(arc_handle);
      }
      arcp2p_transfer(arc_handle, addr, offset, aligned_size, req.p_cpl, dir);
    }
  }

  on_the_fly += 1;

  req_queue.push(req);
}

bool ARC_memory::Arcp2pReqEmpty() {
  return req_queue.empty();
}

// completion
void ARC_memory::Arcp2pCompletion(bool prefCall) {

  // Automatically prefetch
  if(!at::globalContext().ARCGlobal.isOnDemand()) {
    if (pref_end >= pref_idx && prefCall) {
      if (isDebug) {
        std::cout << "Prefetching oid call: " << pref_it[pref_idx] << std::endl;
      }
      bool d2h_finish = torch::autograd::ARCCppEngine::preFetch(pref_it[pref_idx]);
      if (d2h_finish) {
        pref_idx++;
      }
    }
  }

  if(isUsingSSD) {
    // if req_list empty, nothing to do
    if (req_queue.empty()) {
      return;
    }

    // first, run completer of arcp2p, this will update cpl.issued
    arcp2p_completion(arc_handle);

    // we only concern command completion sequentially
    req_element req = req_queue.front();

    if (true == req.p_cpl->issued) {
      // if completed request is ssdtogpu
      // 1. we need to update fetch_loc
      // 2. we should remove loc_element
  
      if (arcp2p_gputossd == req.dir) {
        size_t numel = get_numel(req.info->tid);
        int resize = get_resize(req.info->tid);
        if (isFP16 && (resize > 0)) {
          if (isDebug)
            std::cout << "CSR FP16 mem free tid: " << req.info->tid << ", size: " << sizeof(__half) * resize << ", fp16: " << req.info->ptr << std::endl;
  
          p2p_free(req.info->ptr, sizeof(__half) * resize);
        } else if (isFP16 && (resize == 0)) {
          if (isDebug)
            std::cout << "No CSR FP16 mem free tid: " << req.info->tid << ", size: " << sizeof(__half) * numel << ", fp16: " << req.info->ptr << std::endl;
  
          p2p_free(req.info->ptr, sizeof(__half) * numel);
        } else {
          if (isDebug)
            std::cout << "TODO: Duplicated FP16 mem free tid: " << req.info->tid << ", size: " << req.size << ", fp16: " << req.info->ptr << std::endl;
  
          p2p_free(req.info->ptr, req.size);
        }
  
        event_arr_d2h[req.info->tid] = false;
        delete req.info;
  
        if (false == isFP16)
          delete req.stor;

      } else if (arcp2p_ssdtogpu == req.dir) {
        // [TODO] backend job needed for read done case (ex. notify backward operation that data is ready)
        // [TODO] arcp2p_data would be freed here? or after?
  
        // FP16 & CSR handling
        int resize = get_resize(req.info->tid);
  
        if (isFP16 && (resize > 0)) {
          uint64_t nTPB = req.info->ntpb;
          uint64_t numel = req.info->numel;
  
          size_t bit_elements, pos_elements, pos_elements_before;
          bit_elements = (size_t)((numel + 1024 - 1) / 1024) * 32;
          pos_elements_before = (size_t)((numel + 32 - 1) / 32);
          int count = 0;
          while (pos_elements_before != 0) {
            pos_elements_before = pos_elements_before >> 1;  count++;
          }
          pos_elements = 1 << count;
          
          void* bit = arc_vm.get_bit_addr(req.info->tid);
          void* pos = arc_vm.get_pos_addr(req.info->tid);
          float *nz_dst;
          p2p_malloc((void **)&nz_dst, resize * sizeof(float));
          cudaMemsetAsync((void *)nz_dst, 0, resize * sizeof(float), req.str);
  
          float_scale<<<(numel + nTPB - 1) / nTPB, nTPB, 0, req.str>>>((__half *)req.info->ptr, nz_dst, resize);
//          float_scale<<<(resize + nTPB - 1) / nTPB, nTPB, 0, req.str>>>((__half *)req.info->ptr, nz_dst, resize);
  
          if (arc_vm.get_elem(req.info->tid) == 8) {
            zero_insert_double<<<(numel + nTPB - 1) / nTPB, nTPB, 0, req.str>>>((unsigned int*)bit, (unsigned int*)pos, nz_dst, (double *)req.info->dst, numel);
          } else {
            zero_insert_float<<<(numel + nTPB - 1) / nTPB, nTPB, 0, req.str>>>((unsigned int*)bit, (unsigned int*)pos, nz_dst, (float *)req.info->dst, numel);
          }
  
          p2p_free((void *)nz_dst, resize * sizeof(float));
        } else if (isFP16 && (resize == 0)) {
          uint64_t nTPB = req.info->ntpb;
          uint64_t numel = req.info->numel;
  
          if (arc_vm.get_elem(req.info->tid) == 8) {
            double_scale<<<(numel + nTPB - 1) / nTPB, nTPB, 0, req.str>>>((__half* )req.info->ptr, (double* )req.info->dst, numel);
          } else {
            float_scale<<<(numel + nTPB - 1) / nTPB, nTPB, 0, req.str>>>((__half* )req.info->ptr, (float* )req.info->dst, numel);
          }
        } else {
          cudaMemcpyAsync(req.info->dst, req.info->ptr, req.size, cudaMemcpyDeviceToDevice, req.str);
        }
  
        event_arr_h2d[req.info->tid] = false;
        delete req.info;
      }
  
      on_the_fly -= 1;
      req.p_cpl->requested = false;
  
      // remove current element
      req_queue.pop();
  
      if (false == isTesla) {
        arcp2p_bar_detach(arc_handle);
  
        // check if next event is pending
        if (!req_queue.empty()) {
          req = req_queue.front();
          printf("schedule next one. quadro only\n");
          //arcp2p_bar_attach(arc_handle, req.addr, req.size);
          // debug code. retry 10 times
          int retrycnt = 0;
          while(ARCP2P_NO_ERROR != arcp2p_bar_attach(arc_handle, req.addr, req.size)) {
            retrycnt ++;
            printf("Bar attach failed, retry %d/10\n", retrycnt);
            if (retrycnt >= 10) {
              break;
            }
            arcp2p_bar_detach(arc_handle);
          }
          arcp2p_transfer(arc_handle, req.addr, req.offs, req.size, req.p_cpl, req.dir);
        }
      }
    }
  }
}

void ARC_memory::Arcp2pSynchronize() {
  arcp2p_synchronize(arc_handle);
}

void ARC_memory::kernelTimeStart() {
  if (isTimer) {
    auto str = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard csg(str);
    cudaEventRecord(startEvent, csg.original_stream());
//    cudaDeviceSynchronize();
//    gettimeofday(&tv1, NULL);
  }
}

float* ARC_memory::kernelTimeEnd() {
  if (isTimer) {
    auto str = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard csg(str);
    cudaEventRecord(endEvent, csg.original_stream());
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&runTime, startEvent, endEvent);
  } else {
    runTime = -1;
  }
  return &runTime;
}
}}
