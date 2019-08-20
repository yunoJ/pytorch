#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <mutex>

#include <ATen/native/cuda/arc_flag.h>
#include <ATen/Context.h>
#define NUM_TENSOR 1024

// [JS] P2P define
#include <queue>
#include <c10/cuda/CUDACachingAllocator.h>

// Half precision
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

#define find(n) (32 * (unsigned int)(n / 1024) + (n % 32))
#define mask(n) (0x80000000 >> (unsigned int)((n % 1024) / 32))

// [JS] Arcp2p Setting flag define
#define ARC_FLAG_SSD   (1U << 0)
#define ARC_FLAG_FP16  (1U << 1)
#define ARC_FLAG_CSR   (1U << 2)
#define ARC_FLAG_TESLA (1U << 3)
#define ARC_FLAG_DEBUG (1U << 4)

using namespace at::cuda;
__global__ void float_scale(__half *din, float *dout, int dsize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dsize)  dout[idx] = __half2float(din[idx]);
}

__global__ void zero_insert(unsigned int *bit, unsigned int *nz_pos, float* din, float *dout, int dsize) {
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

ARC_memory arc_vm;

typedef struct {
  uint64_t addr;
  uint64_t size;
  uint64_t offs;
  arcp2p_dir dir;
  arcp2p_cpl *p_cpl;

  c10::Storage *stor;
  arcp2p_info *info;
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

  ARC_memory::ARC_memory(): relu_thru(false), mapping(false), deviceStartBlk(0) {
    cudaMalloc(&deviceAddr, DEVICE_SZ);
    deviceTable = new bool[MAX_DEVICE];
    memset(deviceTable, 0, sizeof(bool) * MAX_DEVICE);

    device_page_map = new unsigned int[MAX_DEVICE];
    for (int i = 0; i < MAX_DEVICE; i++) {
      device_page_map[i] = 0;
    }

    fp16_ptr_arr = new uint64_t[NUM_TENSOR];
    bit_ptr_arr = new uint64_t[NUM_TENSOR];
    pos_ptr_arr = new uint64_t[NUM_TENSOR];
    resize_arr = new unsigned int[NUM_TENSOR];
    cpl_flu_ptr_arr = new uint64_t[NUM_TENSOR];
    cpl_pre_ptr_arr = new uint64_t[NUM_TENSOR];
    offset_arr = new uint64_t[NUM_TENSOR];
    dir_arr = new arcp2p_dir[NUM_TENSOR];

    memset(dir_arr, 0, sizeof(arcp2p_dir) * NUM_TENSOR);

    isTesla = false;
    isUsingSSD = false;
    isFP16 = false;
    isCSR = false;
  }

  ARC_memory::~ARC_memory() {
    cudaFree(deviceAddr);
    delete[] deviceTable;
    delete[] device_page_map;

    delete[] fp16_ptr_arr;
    delete[] bit_ptr_arr;
    delete[] pos_ptr_arr;
    delete[] resize_arr;
    delete[] cpl_flu_ptr_arr;
    delete[] cpl_pre_ptr_arr;
    delete[] offset_arr;
    delete[] dir_arr;

    if (true == isTesla)
    {
      arcp2p_bar_detach(arc_handle);
    }
    arcp2p_release(arc_handle);
  }

  void ARC_memory::device_malloc(void** gpu_ptr, size_t size) {
    int reqBlk = std::ceil((double)size / (double)BLK_SZ);
    int blkCheck = 0;
    int retryCnt = 0;
  
    if (reqBlk == 0) return;
  
    m.lock();
  
    while (true) {
      for (int i = deviceStartBlk; i < MAX_DEVICE; i++) {
        blkCheck += 1;
        if (deviceTable[i]) {
          if (device_page_map[i] == 0) {
            LOG(FATAL) << "device_page_map[" << i << "] is zero, size: " << size << ", " << blkCheck << ", " << reqBlk;
          }
          i += device_page_map[i] - 1;
          blkCheck = 0; deviceStartBlk = i + 1;
          continue;
        }
  
        if (blkCheck == reqBlk) {
          device_page_map[deviceStartBlk] = reqBlk;
          *gpu_ptr = (void* )((size_t)deviceAddr + (deviceStartBlk * BLK_SZ));
  
          for (int i = deviceStartBlk; i < deviceStartBlk + reqBlk; i++) {
            deviceTable[i] = true;
          }
  
          deviceStartBlk += reqBlk;
  
          m.unlock();
          return;
        }
      }
      deviceStartBlk = 0;  blkCheck = 0;
      if (retryCnt++ % 10 == 0) {
        LOG(INFO) << "retryCnt " << retryCnt << ", " << size;
      }

      if (retryCnt > 1000) {
        std::cout << "Out-of-memory in device: " << size;
        exit(1);
      }
    }
  }

  void ARC_memory::device_free(void* addr, size_t size) {
    unsigned int startBlk = ((size_t)addr - (size_t)deviceAddr) / BLK_SZ;
    unsigned int reqBlk = std::ceil((double)size / (double)BLK_SZ);

    m.lock();

    device_page_map[startBlk] = 0;

    for (unsigned int i = startBlk; i < startBlk + reqBlk; i++) {
      deviceTable[i] = false;
    }

    deviceStartBlk = std::min((unsigned int)deviceStartBlk, startBlk);

    m.unlock();
  }

  void* ARC_memory::get_fp16_addr(int tid) {
    return (void *)fp16_ptr_arr[tid];
  }

  void ARC_memory::set_fp16_addr(int tid, uint64_t addr) {
    fp16_ptr_arr[tid] = addr;
  }

  void* ARC_memory::get_device_addr() {
    return deviceAddr;
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

  unsigned int ARC_memory::get_resize(int tid) {
    return resize_arr[tid];
  }

  void ARC_memory::set_resize(int tid, unsigned int resize) {
    resize_arr[tid] = resize;
  }

  void* ARC_memory::get_cpl_addr(int tid, arcp2p_dir dir) {
    if (arcp2p_gputossd == dir)
    {
      return (void *)cpl_flu_ptr_arr[tid];
    }
    else if (arcp2p_ssdtogpu == dir)
    {
      return (void *)cpl_pre_ptr_arr[tid];
    }
    else
    {
      return nullptr;
    }
  }

  void ARC_memory::set_cpl_addr(int tid, arcp2p_dir dir, void *addr) {
    if (arcp2p_gputossd == dir)
    {
      cpl_flu_ptr_arr[tid] = (uint64_t)addr;
    }
    else if (arcp2p_ssdtogpu == dir)
    {
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

  bool ARC_memory::is_using_ssd(void) {
    return isUsingSSD;
  }

  bool ARC_memory::is_fp16(void) {
    return isFP16;
  }

  bool ARC_memory::is_csr(void) {
    return isCSR;
  }

  void ARC_memory::Arcp2pSetting(int flags)
  {
    printf("Arcp2pSetting : 0x%x\n", flags);

    if (flags & ARC_FLAG_FP16)
    {
      printf("FP16 flag set\n");
      isFP16 = true;
    }

    if (flags & ARC_FLAG_CSR)
    {
      printf("CSR flag set\n");
      isCSR = true;
      isFP16 = true;
    }

    if (flags & ARC_FLAG_TESLA)
    {
      printf("Tesla flag set\n");
      isTesla = true;
    }

    if (flags & ARC_FLAG_DEBUG)
    {
      printf("Debug mode on\n");
      at::globalContext().ARCGlobal.turnOnDebugMode();
    }

    if (flags & ARC_FLAG_SSD)
    {
      printf("SSD flag set\n");
      // [JS] P2P
      isUsingSSD = true;
      last_allocated_offset = 0;

      const char *nvme_path_tesla[PATH_LENGTH] = {"0000:65:00.00", "0000:66:00.00"}; // TESLA
      const int nvme_cnt_tesla = 2;
      const char *nvme_path_quadro[PATH_LENGTH] = {"0000:85:00.00", ""}; // QUADRO
      const int nvme_cnt_quadro = 1;

      void* lib_handle;
      if (!(lib_handle = dlopen("/usr/local/lib/libarcp2p.so", RTLD_LAZY)))
      {
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

      if (true == isTesla)
      {
        arc_handle = arcp2p_initialize(nvme_path_tesla, nvme_cnt_tesla);
      }
      else
      {
        arc_handle = arcp2p_initialize(nvme_path_quadro, nvme_cnt_quadro);
      }
    }
    else // if not ssd
    {
      isUsingSSD = false;
    }
  }

  // bar attach
  int  ARC_memory::Arcp2pBarMapping(uint64_t addr, uint64_t size)
  {
    return arcp2p_bar_attach(arc_handle, addr, size);
  }

  // submission
  void ARC_memory::Arcp2pSubmission(uint64_t addr, uint64_t size, uint64_t *p_offs, arcp2p_cpl *p_cpl, arcp2p_dir dir, c10::Storage *stor, arcp2p_info *info)
  {
    uint64_t offset, aligned_size;

    const uint64_t prp_align_size = (1UL << 12);
    const uint64_t prp_align_mask = (prp_align_size - 1);

    // align up the size value
    if (size & prp_align_mask)
    {
      aligned_size = (size + prp_align_size - 1) & (~prp_align_mask);
    }
    else
    {
      aligned_size = size;
    }

    if (arcp2p_gputossd == dir)
    {
      // flush case, need to allocate nvme area
      offset = last_allocated_offset;
      last_allocated_offset = last_allocated_offset + aligned_size;

      *p_offs = offset;
    }
    else
    {
      // prefetch case, handle requested nvme offset
      offset = *p_offs;
    }


    req_element req;
    req.addr = addr;
    req.size = aligned_size;
    req.dir = dir;
    req.stor = stor;
    req.info = info;

    req.offs = offset;
    req.p_cpl = p_cpl;

    req.p_cpl->requested = true;
    req.p_cpl->arc_handle = arc_handle;

    if (true == isTesla)
    {
      // directly deliver transfer request to arcp2p library, only for tesla
      arcp2p_transfer(arc_handle, addr, offset, aligned_size, req.p_cpl, dir);
    }
    else
    {
      // for quadro, we need to attach bar range before transfer
      // check that queue is empty, else case will be handled at completion function
      if (req_queue.empty())
      {
        printf("Transfer directly\n");
        //arcp2p_bar_attach(arc_handle, addr, size);
        // debug code. retry 10 times
        int retrycnt = 0;
        while(ARCP2P_NO_ERROR != arcp2p_bar_attach(arc_handle, addr, size))
        {
          retrycnt ++;
          printf("Bar attach failed, retry %d/10\n", retrycnt);
          if (retrycnt >= 10)
          {
            break;
          }
          arcp2p_bar_detach(arc_handle);
        }
        arcp2p_transfer(arc_handle, addr, offset, aligned_size, req.p_cpl, dir);
      }
    }

    req_queue.push(req);
  }

  // completion
  void ARC_memory::Arcp2pCompletion()
  {
    // if req_list empty, nothing to do
    if (req_queue.empty())
    {
      return;
    }

    // first, run completer of arcp2p, this will update cpl.issued
    arcp2p_completion(arc_handle);

    // we only concern command completion sequentially
    req_element req = req_queue.front();

    if (true == req.p_cpl->issued)
    {
      req.p_cpl->requested = false;

      // if completed request is ssdtogpu
      // 1. we need to update fetch_loc
      // 2. we should remove loc_element
      if (arcp2p_ssdtogpu == req.dir)
      {
//        printf("NVMe read done\n");
        // [TODO] backend job needed for read done case (ex. notify backward operation that data is ready)
        // [TODO] arcp2p_data would be freed here? or after?

        // FP16 & CSR handling
        unsigned int resize = arc_vm.get_resize(req.info->tid);

        if (isFP16 && (resize > 0))
        {
          auto stream = at::cuda::getCurrentCUDAStream();
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
          device_malloc((void **)&nz_dst, resize * sizeof(float));
          cudaMemsetAsync((void *)nz_dst, 0, resize * sizeof(float), stream);

          float_scale<<<(numel + nTPB - 1) / nTPB, nTPB, 0, stream>>>((__half *)req.info->ptr, nz_dst, resize);
          zero_insert<<<(numel + nTPB - 1) / nTPB, nTPB, 0, stream>>>((unsigned int*)bit, (unsigned int*)pos, nz_dst, (float *)req.info->dst, numel);

          device_free((void *)nz_dst, resize * sizeof(float));
          device_free(req.info->ptr, sizeof(__half) * resize);
          device_free(bit, sizeof(unsigned int) * bit_elements);
          device_free(pos, sizeof(unsigned int) * pos_elements);

          delete req.info;
        }
        else if (isFP16 && (resize == 0))
        {
          auto stream = at::cuda::getCurrentCUDAStream();

          uint64_t nTPB = req.info->ntpb;
          uint64_t numel = req.info->numel;

          float_scale<<<(numel + nTPB - 1) / nTPB, nTPB, 0, stream>>>((__half* )req.info->ptr, (float* )req.info->dst, numel);
          device_free(req.info->ptr, sizeof(__half) * numel);

          delete req.info;
        }
      }
      else
      {
//        printf("NVMe write done\n");
        // [TODO] backend job needed for write done case (ex. gpu memory free)
        // Because of constructor / destructor definition of storage_impl in Tensor,
        // We just drop Tensor at here
        delete req.stor;
      }

      // remove current element
      req_queue.pop();

      if (false == isTesla)
      {
        arcp2p_bar_detach(arc_handle);

        // check if next event is pending
        if (!req_queue.empty())
        {
          req = req_queue.front();
          printf("schedule next one. quadro only\n");
          //arcp2p_bar_attach(arc_handle, req.addr, req.size);
          // debug code. retry 10 times
          int retrycnt = 0;
          while(ARCP2P_NO_ERROR != arcp2p_bar_attach(arc_handle, req.addr, req.size))
          {
            retrycnt ++;
            printf("Bar attach failed, retry %d/10\n", retrycnt);
            if (retrycnt >= 10)
            {
              break;
            }
            arcp2p_bar_detach(arc_handle);
          }
          arcp2p_transfer(arc_handle, req.addr, req.offs, req.size, req.p_cpl, req.dir);
        }
      }
    }
  }

}}
