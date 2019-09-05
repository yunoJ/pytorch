#include <ATen/Config.h>

#include <ATen/Context.h>

#include <c10/core/TensorOptions.h>

#include <thread>
#include <mutex>
#include <sstream>
#include <string>
#include <stdexcept>

#include <ATen/Tensor.h>
#include <ATen/cpu/FlushDenormal.h>

#include <TH/TH.h>  // for USE_LAPACK
#include <ATen/native/cuda/arc_flag.h>

namespace at {

Context::Context()
: thc_state(nullptr, [](THCState* p){ /* no-op */ } )
, thh_state(nullptr, [](THHState* p){ /* no-op */ } ) {}

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
Context & globalContext() {
  static Context globalContext_;
  return globalContext_;
}

// NB: This method is *purely* whether or not a user requested
// that CuDNN was enabled, it doesn't actually say anything about
// whether or not CuDNN is actually usable.
bool Context::userEnabledCuDNN() const {
  return enabled_cudnn;
}

void Context::setUserEnabledCuDNN(bool e) {
  enabled_cudnn = e;
}

bool Context::deterministicCuDNN() const {
  return deterministic_cudnn;
}

void Context::setDeterministicCuDNN(bool b) {
  deterministic_cudnn = b;
}

bool Context::benchmarkCuDNN() const {
  return benchmark_cudnn;
}

void Context::setBenchmarkCuDNN(bool b) {
  benchmark_cudnn = b;
}

bool Context::hasMKL() const {
#if AT_MKL_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasMKLDNN() const {
#if AT_MKLDNN_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasOpenMP() const {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

bool Context::hasLAPACK() const {
#ifdef USE_LAPACK
  return true;
#else
  return false;
#endif
}

bool Context::setFlushDenormal(bool on) {
  return at::cpu::set_flush_denormal(on);
}

Allocator* getCPUAllocator() {
  return getTHDefaultAllocator();
}

struct LegacyDeviceTypeInit : public LegacyDeviceTypeInitInterface {
  LegacyDeviceTypeInit(LegacyDeviceTypeInitArgs) {}
  void initCPU() const override {
    globalContext();
  }
  void initCUDA() const override {
    globalContext().lazyInitCUDA();
  }
  void initHIP() const override {
    globalContext().lazyInitHIP();
  }
};
REGISTER_LEGACY_TYPE_INIT(LegacyDeviceTypeInit);

/////////////////////////////////////////////////////
// Implemented by SNU-ARC Function/Data Structures///
// //////////////////////////////////////////////////

#define BP_NUM_PER_ITER 1
#define RESET_TID 0 // 4-4 = 0

// network
static bool cycle_gan = 0;
static bool bert = 1;

bool Context::ARCGlobalContext::isCycleGAN() {return cycle_gan;}
bool Context::ARCGlobalContext::isBERT() {return bert;}


// Data structure definitions
// tensor, operation counts
static int global_tensor_id_ = 0;
static int global_operation_id_ = 0;
// flags  
// on_demand_mode is required to construct back_path_
static bool on_debug_mode_ = 0;
static bool on_demand_mode_ = 1; // default 1. Set 0 after first iteration(Profiling Stage) 
static bool on_forwarding_ = 1; // 1 in forwarding phase. 0 in backprop. phase
// vector for prefetching
//Note: C++ standard containers are thread-safe.
//static std::vector<Oid> back_path_[BP_NUM_PER_ITER];
static Oid back_path_[BP_NUM_PER_ITER][NUM_OP] = {0};
static int back_path_idx[BP_NUM_PER_ITER] = {-1};

static int cur_back_num = 0;
//offload prefetch stream
static auto offload_stream = c10::cuda::getStreamFromPool(false, 0);
static auto prefetch_stream = offload_stream;//c10::cuda::getStreamFromPool(false, 0);

int Context::ARCGlobalContext::curBackNum() { return cur_back_num; }

// tid, oid manipulation
c10::cuda::CUDAStream Context::ARCGlobalContext::globalOffloadStream() { return offload_stream; }
c10::cuda::CUDAStream Context::ARCGlobalContext::globalPrefetchStream() { return prefetch_stream; }

Tid Context::ARCGlobalContext::getTid(Tensor& t) { return t.unsafeGetTensorImpl()->tensor_id; }

void Context::ARCGlobalContext::setNewTid(Tensor& t) { 
    t.unsafeGetTensorImpl()->tensor_id = ++global_tensor_id_; 
}
void Context::ARCGlobalContext::updateTid(Tensor& t, int tid) { t.unsafeGetTensorImpl()->tensor_id = tid; }
void Context::ARCGlobalContext::resetGlobalTid() { 
    if (cycle_gan) global_tensor_id_ = RESET_TID;
    else global_tensor_id_ = 0; 
}
Oid Context::ARCGlobalContext::getCurOid() { return global_operation_id_; }
Oid Context::ARCGlobalContext::getNewOid() { return ++global_operation_id_; }
void Context::ARCGlobalContext::resetGlobalOid() { global_operation_id_ = 0; }

// set flags
void Context::ARCGlobalContext::startForward() { 
    on_forwarding_ = 1; 
    cur_back_num++;
    if(cur_back_num == BP_NUM_PER_ITER)
        cur_back_num = 0;
}
void Context::ARCGlobalContext::endForward() { on_forwarding_ = 0; }
void Context::ARCGlobalContext::endOnDemand() { 
    static int remaining_backward_in_first_iter = BP_NUM_PER_ITER;
    --remaining_backward_in_first_iter;
    if (remaining_backward_in_first_iter == 0)
        on_demand_mode_ = 0;
}
// flag checks 
bool Context::ARCGlobalContext::isForward() { return on_forwarding_; }
bool Context::ARCGlobalContext::isOnDemand() { return on_demand_mode_; }
bool Context::ARCGlobalContext::isDebugMode() { return on_debug_mode_; }
void Context::ARCGlobalContext::turnOnDebugMode() { on_debug_mode_ = 1; }

void Context::ARCGlobalContext::pushBackOid(Oid oid) { 
  if (!on_demand_mode_) std::cerr << "Illegal call: not on-demand mode" << std::endl;

  int idx = ++back_path_idx[cur_back_num];
  back_path_[cur_back_num][idx] = oid;
}

//std::vector<Oid> Context::ARCGlobalContext::getBackPath() {
int* Context::ARCGlobalContext::getBackPath() {
  return back_path_[cur_back_num];
}; 

int Context::ARCGlobalContext::getLastIdx() {
  return back_path_idx[cur_back_num];
};

}
