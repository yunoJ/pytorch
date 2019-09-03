#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/anomaly_mode.h>

#include <ATen/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>
#include <sstream>

//SNU-ARC
#include <c10/core/TensorOptions.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <thread>
#include <queue>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <THC/THCGeneral.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/cuda/arc_flag.h>
#include <sys/time.h>

namespace torch { namespace autograd {

SavedVariable::SavedVariable(const Variable& variable, bool is_output) {
  if (variable.defined()) {
    was_default_constructed_ = false;
    output_nr_ = variable.output_nr();
    requires_grad_ = variable.requires_grad();
    has_grad_fn_ = !variable.is_leaf();
    // These copies are all shared_ptr copies, so slightly more expensive.
    // Do them here instead of in the init list in case data is undefined.
    data_ = variable.tensor_data();
    if (variable.is_leaf()) {
      grad_accumulator_ = variable.grad_accumulator();
    } else if (!is_output) {
      grad_fn_ = variable.grad_fn();
    }
    version_counter_ = variable.version_counter();
    saved_version_ = version_counter_.current_version();
  }
}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Variable();
  }

  auto grad_fn = grad_fn_;
  if (has_grad_fn_ && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    grad_fn = std::move(saved_for);
  }

  if (saved_version_ != version_counter_.current_version()) {
    std::stringstream message;
    message << "one of the variables needed for gradient computation has been "
        "modified by an inplace operation: [" << data_.type().toString() << " "
        << data_.sizes() << "]";
    if (grad_fn) {
        message << ", which is output " << output_nr_
            << " of " << grad_fn->name() << ",";
    }
    message << " is at version " << version_counter_.current_version()
        << "; expected version " << saved_version_ << " instead.";
    if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
            "that failed to compute its gradient, with torch.autograd."
            "set_detect_anomaly(True).";
    }
    else {
        message << " Hint: the backtrace further above shows the operation "
            "that failed to compute its gradient. The variable in question "
            "was changed in there or anywhere later. Good luck!";
    }
    throw std::runtime_error(message.str());
  }

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data_, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data_, requires_grad_);
  }
  var.set_version_counter(saved_version_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph).
  if (requires_grad_ && !var.grad_fn() && grad_accumulator_.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  var.set_grad_accumulator(grad_accumulator_);

  return var;
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the buffers have "
    "already been freed. Specify retain_graph=True when calling backward "
    "the first time.";

/////////////////////////////////////////////////////
// Implemented by SNU-ARC Function/Data Structures///
// //////////////////////////////////////////////////

// thread vector(it may cause system slower. Need an optimization such as pooling) 
static std::vector<std::thread> memcpy_threads_;
// dictionaries for prefetching
// Note: C++ standard containers are thread-safe.
static std::map<Oid,std::vector<PFInfo>> pf_dict_;
static std::map<Oid, c10::StreamId> pf_sync_dict_;

static at::Tensor tensor_dict_[NUM_TENSOR];
static bool tensor_dict_check_[NUM_TENSOR] = {false};
static bool tensor_pf_sync_dict_[NUM_TENSOR] = {false};

static bool tensor_sync_dict_[NUM_TENSOR] = {false};
static std::map<Tid, Oid> last_op_dict_;

static std::map<Tid, std::pair<double, bool>> liveness_temp;

static bool liveness_result[NUM_TENSOR] = {false};
static bool liveness_result_csr[NUM_TENSOR] = {false};

// thread for prefetch
//static std::thread prefetch_thread_;

static double accumSize = 0;

double ARCCppEngine::checkCSR(double freeSize) {
  double remainSize = accumSize - freeSize;
  std::cout << "checkCSR" << std::endl;

  if (remainSize <= 0) {
    return 0;
  }

  for (auto it = liveness_temp.begin(); it != liveness_temp.end(); ++it) {
    if ((it->second).second) {
      if (at::globalContext().ARCGlobal.isDebugMode()) {
        std::cout << "\tcheckCSR tid: " << it->first << ", size: " << (it->second).first << std::endl;
      }
      remainSize -= (it->second).first;
      liveness_result[it->first] = true;
      liveness_result_csr[it->first] = (it->second).second;
    }

    if (remainSize <= 0) {
      break;
    }
  }

  return remainSize;
}

double ARCCppEngine::checkLarge(double remainSize) {
  for (auto it = liveness_temp.begin(); it != liveness_temp.end(); ++it) {
    if (liveness_result[it->first] == false) {
      if ((it->second).first > 12) {
        if (at::globalContext().ARCGlobal.isDebugMode()) {
          std::cout << "\tcheckLarge tid: " << it->first << ", size: " << (it->second).first << std::endl;
        }
        remainSize -= (it->second).first;
        liveness_result[it->first] = true;
        liveness_result_csr[it->first] = (it->second).second;
      }
    }

    if (remainSize <= 0)  break;
  }

  return remainSize;
}

double ARCCppEngine::checkFirst(double remainSize) {
  for (auto it = liveness_temp.begin(); it != liveness_temp.end(); ++it) {
    if (liveness_result[it->first] == false) {
      if (at::globalContext().ARCGlobal.isDebugMode()) {
        std::cout << "\tcheckFirst tid: " << it->first << ", size: " << (it->second).first << std::endl;
      }
      remainSize -= (it->second).first;
      liveness_result[it->first] = true;
      liveness_result_csr[it->first] = (it->second).second;
    }

    if (remainSize <= 0)  break;
  }
  return remainSize;
}

//Note: Not referecne but copy a tensor to make it alive
void ARCCppEngine::offLoad(at::Tensor t, /*TraceableFunction* grad_fn, ARCSync sync,*/ Oid curOid, SavedVariable* fetch_loc, bool isOutput) {

  if (!at::native::arc_vm.is_vdnn()) {
    *fetch_loc = SavedVariable(t, isOutput);
    return;
  }

  // partial offloading
  auto tid =  t.unsafeGetIntrusivePtr()->tensor_id;

  if (liveness_result[tid] == false && !at::globalContext().ARCGlobal.isOnDemand()) {
    *fetch_loc = SavedVariable(t, isOutput);
    return;
  }

  if (tid == 0) {
    *fetch_loc = SavedVariable(t, isOutput);
    return;
  }

  insertToPFDict_(curOid, fetch_loc, tid);
  // save the information required for prefetching
  if (tensor_sync_dict_[tid]) {// this tensor is alredy offloaded
    if (at::globalContext().ARCGlobal.isDebugMode())
      std::cout << tid <<  ": this tensor is already offloaded" << std::endl;

    at::native::arc_vm.relu_thru = false;
    return;
  }

  if (at::globalContext().ARCGlobal.isOnDemand() && at::globalContext().ARCGlobal.isForward()) {
    accumSize += (double)t.nbytes() / 1024 / 1024;

    if (at::globalContext().ARCGlobal.isDebugMode()) {
      std::cout << "accumulated tensor tid: " << tid << ", accum: " << accumSize << ", relu_thru: " << at::native::arc_vm.relu_thru << std::endl;
    }

    liveness_temp.insert(std::pair<Tid, std::pair<double, bool>>(
        tid, std::pair<double, bool>((double)t.nbytes() / 1024 / 1024, at::native::arc_vm.relu_thru)));
    at::native::arc_vm.relu_thru = false;
  }

  tensor_sync_dict_[tid] = true;
  offLoadAsync(t);
}

void ARCCppEngine::joinOffload() {
  if (at::globalContext().ARCGlobal.isDebugMode())
    std::cout << "Wait until all offloading is done" << std::endl;

/*
  if (at::native::arc_vm.is_using_ssd()) {
    at::native::arc_vm.Arcp2pSynchronize();
    int count = 1;
    while(count > 0) {
      count = 0;
      for (int i = 0; i < NUM_TENSOR; i++) {
        count += (int)at::native::arc_vm.event_arr_d2h[i];
        at::native::arc_vm.Arcp2pCompletion(false);
      }
    }
  }
*/

  if (at::globalContext().ARCGlobal.isDebugMode())
    std::cout << "Wait end" << std::endl;
}

void ARCCppEngine::offLoadAsync(at::Tensor tensor) {
  auto str = c10::cuda::getStreamFromPool(false, 0);
//  str.synchronize();
  c10::cuda::CUDAStreamGuard csg(str);
  c10::TensorOptions opt = c10::TensorOptions();
  opt = opt.device(c10::Device(c10::DeviceType::CPU));
  opt = opt.dtype(tensor.dtype());
  opt = opt.pinned_memory(true);

  auto tid = tensor.unsafeGetIntrusivePtr()->tensor_id;

  // [JS] p2p setting if ssd mode is on
  if (at::native::arc_vm.is_using_ssd()) {
    at::native::arc_vm.set_dir(tid, at::native::arcp2p_gputossd);
    at::native::arcp2p_cpl *p_cpl = new at::native::arcp2p_cpl;
    at::native::arc_vm.set_cpl_addr(tid, at::native::arcp2p_gputossd, (void *)p_cpl);
  }

  if (at::globalContext().ARCGlobal.isOnDemand()) {
    tensor_dict_[tid] = tensor.ARCto(opt, false, true, false);
    tensor_dict_check_[tid] = true;
  } else {
    tensor_dict_[tid] = tensor.ARCto(opt, false, true, liveness_result_csr[tid]);
    tensor_dict_check_[tid] = true;
  }

  if (at::native::arc_vm.is_using_ssd())
    at::native::arc_vm.Arcp2pCompletion(false);

  csg.reset_stream(csg.original_stream());
}

// doing nothing really
void ARCCppEngine::explicitAllSync() {
  for (auto it = memcpy_threads_.begin(); it != memcpy_threads_.end(); ++it)
    it->join();

  memcpy_threads_.clear();
}

// prefetch all tensors required for back prop of curOid
void ARCCppEngine::preFetch(Oid curOid, ARCSync sync) {//int required_tensor_num, ARCSync sync) {
  if (curOid == 0)
    return;

  if (!at::native::arc_vm.is_vdnn()) {
    return;
  }

//  at::globalContext().ARCGlobal.globalOffloadStream().synchronize();   
  
  bool notused = fetchRequiredTensors_(curOid, sync); 
}

bool ARCCppEngine::preFetchAsync(Oid curOid) {//int required_tensor_num, ARCSync sync) {
  //if (target < 0) std::cerr << "There is no more operation to need prefetch." << std::endl;
  return fetchRequiredTensors_(curOid, Sync); 
}

void ARCCppEngine::preFetchSync(Oid oid, bool isOutput) { 
  //this operation has nothing to prefetch 
  if (oid == 0)
    return;

  if (!at::native::arc_vm.is_vdnn()) {
    return;
  }

  if (pf_dict_.find(oid) == pf_dict_.end()) {
    return;
  }

  while (1) {
    auto check = pf_sync_dict_.find(oid);
    if (check != pf_sync_dict_.end())
      break;
  }

  auto sid = pf_sync_dict_[oid];
  c10::cuda::CUDAStream str(c10::Stream(c10::Stream::UNSAFE, c10::Device(c10::DeviceType::CUDA, 0), sid));
  str.synchronize();

  auto fetch_vec = pf_dict_[oid];
  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;
    auto fetch_loc = it->first;

    if (tensor_pf_sync_dict_[tid] == true) {
      *fetch_loc = SavedVariable(tensor_dict_[tid], isOutput);
      continue;
    }

    if (at::globalContext().ARCGlobal.isDebugMode())
      std::cout << "tid " << tid << " in oid " << oid << " pf sync start" << std::endl;

    if (tensor_dict_check_[tid] == false) {
      std::cerr << "sync tensor dictionary lookup miss" << std::endl;
      return;
    }

    while (1) {
      // [JS] for p2p support
      volatile at::native::arcp2p_cpl *p_flu_cpl = (volatile at::native::arcp2p_cpl *)at::native::arc_vm.get_cpl_addr(tid, at::native::arcp2p_gputossd);
      volatile at::native::arcp2p_cpl *p_pre_cpl = (volatile at::native::arcp2p_cpl *)at::native::arc_vm.get_cpl_addr(tid, at::native::arcp2p_ssdtogpu);

      void* fp16 = at::native::arc_vm.get_fp16_addr(tid);
      size_t numel = at::native::arc_vm.get_numel(tid);

      if (at::native::arc_vm.is_using_ssd()) {
//        if (false == p_flu_cpl->requested && false == p_pre_cpl->requested) {
        if (p_pre_cpl != NULL && false == p_pre_cpl->requested) {
          int resize = at::native::arc_vm.get_resize(tid);
          size_t bit_elements, pos_elements, pos_elements_before;
          bit_elements = (size_t)((numel + 1024 - 1) / 1024) * 32;
          pos_elements_before = (size_t)((numel + 32 - 1) / 32);
          int count = 0;
          while (pos_elements_before != 0) {
            pos_elements_before = pos_elements_before >> 1;  count++;
          }
          pos_elements = 1 << count;

          cudaStreamSynchronize(str);

          if (at::native::arc_vm.is_csr() && resize > 0) {
            if (at::globalContext().ARCGlobal.isDebugMode())
              std::cout << "CSR h2d mem free tid: " << tid << ", size: " << sizeof(__half) * resize << ", fp16: " << fp16 << std::endl;

            void* bit = at::native::arc_vm.get_bit_addr(tid);
            void* pos = at::native::arc_vm.get_pos_addr(tid);

            at::native::arc_vm.p2p_free(fp16, sizeof(__half) * resize);
            at::native::arc_vm.p2p_free(bit, sizeof(unsigned int) * bit_elements);
            at::native::arc_vm.p2p_free(pos, sizeof(unsigned int) * pos_elements);
          } else if (at::native::arc_vm.is_fp16() && resize == 0) {
            // TODO arcp2p without any comprression (fp16 == false, is_csr == false)
            if (at::globalContext().ARCGlobal.isDebugMode())
              std::cout << "No CSR h2d mem free tid: " << tid << ", size: " << sizeof(__half) * numel << ", fp16: " << fp16 << std::endl;

            at::native::arc_vm.p2p_free(fp16, sizeof(__half) * numel);
          } else {
            if (at::globalContext().ARCGlobal.isDebugMode())
              std::cout << "No float/double h2d mem free tid: " << tid << ", size: " << numel << ", fp16: " << fp16 << std::endl;

            at::native::arc_vm.p2p_free(fp16, numel);
          }

          delete p_flu_cpl;
          delete p_pre_cpl;

          at::native::arc_vm.set_cpl_addr(tid, at::native::arcp2p_gputossd, NULL);
          at::native::arc_vm.set_cpl_addr(tid, at::native::arcp2p_ssdtogpu, NULL);

          tensor_pf_sync_dict_[tid] = true;
          break;
        }
        at::native::arc_vm.Arcp2pCompletion(true);
      } else {
        if (tensor_dict_[tid].device().type() == c10::DeviceType::CUDA) {
          int resize = at::native::arc_vm.get_resize(tid);
          size_t bit_elements, pos_elements, pos_elements_before;
          bit_elements = (size_t)((numel + 1024 - 1) / 1024) * 32;
          pos_elements_before = (size_t)((numel + 32 - 1) / 32);
          int count = 0;
          while (pos_elements_before != 0) {
            pos_elements_before = pos_elements_before >> 1;  count++;
          }
          pos_elements = 1 << count;

          if (fp16 != NULL) {
            if (at::native::arc_vm.is_csr() && resize > 0) {
              void* bit = at::native::arc_vm.get_bit_addr(tid);
              void* pos = at::native::arc_vm.get_pos_addr(tid);
              unsigned int resize = at::native::arc_vm.get_resize(tid);

              if (at::globalContext().ARCGlobal.isDebugMode()) {
                std::cout << "CSR h2d mem free tid: " << tid << ", size: " << sizeof(__half) * resize << std::endl;
              }

              at::native::arc_vm.p2p_free(fp16, sizeof(__half) * resize);
              at::native::arc_vm.p2p_free(bit, sizeof(unsigned int) * bit_elements);
              at::native::arc_vm.p2p_free(pos, sizeof(unsigned int) * pos_elements);
            } else if (at::native::arc_vm.is_fp16() && resize == 0) {
              if (at::globalContext().ARCGlobal.isDebugMode()) {
                std::cout << "No CSR h2d mem free tid: " << tid << ", size: " << sizeof(__half) * numel << std::endl;
              }

              at::native::arc_vm.p2p_free(fp16, sizeof(__half) * numel);
            }
          } else {
            if (at::globalContext().ARCGlobal.isDebugMode()) {
              std::cout << "No float/double h2d mem free tid: " << tid << std::endl;
            }
          }

          tensor_pf_sync_dict_[tid] = true;
          at::native::arc_vm.event_arr_h2d[tid] = false;
          break;
        }
      }
    }

    *fetch_loc = SavedVariable(tensor_dict_[tid], isOutput);
  }
}

void ARCCppEngine::insertToPFDict_(Oid oid, SavedVariable* loc, Tid tid) {
  //lookup and insert pf_dict_
  auto it = pf_dict_.find(oid);
  if (it != pf_dict_.end()) {
    pf_dict_[oid].emplace_back(loc, tid);
  } else {
    std::vector<PFInfo> tmp;
    tmp.emplace_back(loc, tid);
    pf_dict_.insert(std::pair<Oid, std::vector<PFInfo>>(oid, tmp));
  }
}

// drop tensor from gpu memory
// this function should be called at the end of back prop of each layer
void ARCCppEngine::dropTensor(Oid oid, SavedVariable* fetch_loc) { 
  //this operation has nothing to prefetch 
  if (pf_dict_.find(oid) == pf_dict_.end()) {
    //std::cerr << oid << " Prefetching dictionary lookup miss" << std::endl;
    return;
  }

  auto fetch_vec = pf_dict_[oid];
  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;
    if (at::globalContext().ARCGlobal.isOnDemand()) {
      at::Tensor& tref = tensor_dict_[tid]; 
      c10::TensorOptions opt = c10::TensorOptions();
      opt = opt.device(c10::Device(c10::DeviceType::CPU));
      opt = opt.dtype(tref.dtype());
      //opt = opt.dtype(c10::ScalarType::Half); 
      opt = opt.pinned_memory(true); 

      while (at::native::arc_vm.event_arr_h2d[tid]) {
        if (at::native::arc_vm.is_using_ssd()) {
          at::native::arc_vm.Arcp2pCompletion(false);
        }
      }

      // [JS] p2p setting if ssd mode is on
      if (at::native::arc_vm.is_using_ssd()) {
        at::native::arc_vm.set_dir(tid, at::native::arcp2p_gputossd);
        at::native::arcp2p_cpl *p_cpl = new at::native::arcp2p_cpl;
        at::native::arc_vm.set_cpl_addr(tid, at::native::arcp2p_gputossd, (void *)p_cpl);
      }

      tref = tref.ARCto(opt, false, true, false);
      tensor_pf_sync_dict_[tid] = false;

      while (at::native::arc_vm.event_arr_d2h[tid]) {
        if (at::native::arc_vm.is_using_ssd()) {
          at::native::arc_vm.Arcp2pCompletion(false);
        }
      }
    } else {
      if ((oid == last_op_dict_[tid]) && tensor_dict_check_[tid]) {
        tensor_dict_check_[tid] = false;
        fetch_loc->reset_data();
      }
    }
  }
}

bool ARCCppEngine::fetchRequiredTensors_(Oid oid, ARCSync sync) {
  //this operation has nothing to prefetch 
  if (pf_dict_.find(oid) == pf_dict_.end()) {
    //std::cerr << oid << " Prefetching dictionary lookup miss" << std::endl;
    return true;
  }

  if (at::globalContext().ARCGlobal.isOnDemand()) {
    at::globalContext().ARCGlobal.pushBackOid(oid);
  }
  
  auto fetch_vec = pf_dict_[oid];

  auto str = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard csg(str);
  pf_sync_dict_.insert(std::pair<Oid, c10::StreamId>(oid, str.id()));

  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;

    if (tensor_dict_check_[tid] == false) {
      //std::cerr << "tensor dictionary lookup miss" << std::endl;
      return true;
    }

    at::Tensor& tref = tensor_dict_[tid];
    c10::TensorOptions opt = c10::TensorOptions();
    opt = opt.device(c10::Device(c10::DeviceType::CUDA));
    opt = opt.dtype(tref.dtype());
    //opt = opt.dtype(c10::ScalarType::Float);
 
    if (tref.device().type() == c10::DeviceType::CPU) {

      // [JS] p2p
      if (at::native::arc_vm.is_using_ssd()) {
        if (at::globalContext().ARCGlobal.isOnDemand()) {
          while (at::native::arc_vm.event_arr_d2h[tid]) {
            at::native::arc_vm.Arcp2pCompletion(false);
          }
        } else {
          if (at::native::arc_vm.event_arr_d2h[tid]) {
            return false;
          }
        }
        at::native::arc_vm.set_dir(tid, at::native::arcp2p_ssdtogpu);
        at::native::arcp2p_cpl *p_cpl = new at::native::arcp2p_cpl;
        p_cpl->requested = true;
        at::native::arc_vm.set_cpl_addr(tid, at::native::arcp2p_ssdtogpu, (void *)p_cpl);
      }

      if (at::globalContext().ARCGlobal.isOnDemand()) {
        if (last_op_dict_.find(tid) == last_op_dict_.end()) {
          last_op_dict_.insert(std::pair<Tid,Oid>(tid, oid));
        } else {
          last_op_dict_[tid] = oid;   
        }
      }

      if (at::globalContext().ARCGlobal.isOnDemand()) {
        tref = tref.ARCto(opt, false, true, false);
      } else {
        tref = tref.ARCto(opt, false, true, liveness_result_csr[tid]);
      }
    } else {
      if (at::globalContext().ARCGlobal.isDebugMode())
        std::cout << tid << ": This tensor is already fetched" << std::endl;
    }
  }
  return true;
}

void ARCCppEngine::resetCppEngine() {
  static int backward_num_CycleGAN = 3;
  static int backward_num_BERT = 1;
  static int remaining_backward = -1;//backward_num_in_one_iter;

  if (remaining_backward == -1) {
    if (at::globalContext().ARCGlobal.isCycleGAN()) {
        remaining_backward = backward_num_CycleGAN;
    } else {
//    if (at::globalContext().ARCGlobal.isBERT())
        remaining_backward = backward_num_BERT;
    }
  }
  

  for(auto i = 0; i < NUM_TENSOR; i ++) {
    if (tensor_pf_sync_dict_[i] == true) {
      tensor_dict_[i].reset();
    }
  }

  memset(tensor_dict_check_, 0, sizeof(bool) * NUM_TENSOR);
  memset(tensor_sync_dict_, 0, sizeof(bool) * NUM_TENSOR);
  memset(tensor_pf_sync_dict_, 0, sizeof(bool) * NUM_TENSOR);
  pf_dict_.clear();
  pf_sync_dict_.clear();

  --remaining_backward;
  if (remaining_backward == 0) {
    at::globalContext().ARCGlobal.resetGlobalTid();
    at::globalContext().ARCGlobal.resetGlobalOid();
    if (at::globalContext().ARCGlobal.isCycleGAN()) {
        remaining_backward = backward_num_CycleGAN;
    } else {
//    if (at::globalContext().ARCGlobal.isBERT())
        remaining_backward = backward_num_BERT;
    }
  }
}

}} // namespace torch::autograd
