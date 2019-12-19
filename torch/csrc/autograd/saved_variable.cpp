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

#include <c10/cuda/CUDACachingAllocator.h>

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

// dictionaries for prefetching
// Note: C++ standard containers are thread-safe.
static std::map<Oid, std::vector<PFInfo>> op_tensor_list;
static std::map<Oid, c10::StreamId> stream_occupied;

static at::Tensor target_tensor[NUM_TENSOR];
static bool target_tensor_valid[NUM_TENSOR] = {false};

static bool offload_sync[NUM_TENSOR] = {false};
static bool prefetch_sync[NUM_TENSOR] = {false};
static int last_use_forward[3][NUM_TENSOR] = {-1};
static int last_use_backward[3][NUM_TENSOR] = {-1};

static double liveness_time[NUM_TENSOR] = {0.0};
static double liveness_size[NUM_TENSOR] = {0.0};
static bool liveness_csr[3][NUM_TENSOR] = {false};
static bool liveness_fp[3][NUM_TENSOR] = {false};

static double last_time_slot = 0;

// thread for prefetch
//static std::thread prefetch_thread_;

static double accumSize = 0;
static double ssd_w = 3072;
static double ssd_r = 10240;
static double mem_wr = 12288;

void ARCCppEngine::checkTest2(double freeSize) {
  double accumTime = 0;
  double delay_time[NUM_TENSOR] = {0};
  double real_trans_time[NUM_TENSOR] = {0};
  double real_trans_start[NUM_TENSOR] = {0};
  double remainSize = accumSize - freeSize;
  accumSize = 0;
  int cur_back_num = at::globalContext().ARCGlobal.curBackNum();

  for (int i = 0; i < NUM_TENSOR-1; i++)
    liveness_time[i + 1] += liveness_time[i];

  last_time_slot += liveness_time[NUM_TENSOR - 1];

  if (at::globalContext().ARCGlobal.isDebugMode()) {
    std::cout << "Possible resoure time: " << last_time_slot << ", mem: " << freeSize << ", remained size: " << remainSize << std::endl;

    if (last_time_slot / 1000 * ssd_w > remainSize) {
      std::cout << "\tNo overhead maybe" << std::endl;
    } else {
      std::cout << "\tOverhead maybe" << std::endl;
    }
  }

  for (int i = 0; i < NUM_TENSOR; i++) {
    if (at::native::arc_vm.is_using_ssd())
      real_trans_time[i] = liveness_size[i] * 1000 / ssd_w;
    else
      real_trans_time[i] = liveness_size[i] * 1000 / mem_wr;

    if (at::native::arc_vm.is_fp16() && liveness_fp[cur_back_num][i])
      real_trans_time[i] = real_trans_time[i] / 2;

    if (at::native::arc_vm.is_csr() && liveness_csr[cur_back_num][i])
      real_trans_time[i] = real_trans_time[i] / 2;
  }

  if (remainSize <= 0) {
    std::cout << "Nothing" << std::endl;
    return;
  }

  int previous_i = 0;

  for (int i = 0; i < NUM_TENSOR; i++) {
    if (liveness_size[i] > 1) {
      if (at::globalContext().ARCGlobal.isDebugMode())
        std::cout << "Test tid: " << i << ", size: " << liveness_size[i] << ", time: " << liveness_time[i] << std::endl;

      at::native::arc_vm.liveness_result[cur_back_num][i] = true;
      remainSize -= liveness_size[i];

      previous_i = i;
      break;
    }
  }

  for (int i = previous_i + 1; i < NUM_TENSOR; i++) {
    if (liveness_size[i] > 1) {
      at::native::arc_vm.liveness_result[cur_back_num][i] = true;
      remainSize -= liveness_size[i];

      double delay_maybe = liveness_time[previous_i] + real_trans_time[previous_i] + delay_time[previous_i] - liveness_time[i];

      if (delay_maybe <= 0)
        delay_time[i] = 0;
      else
        delay_time[i] = delay_maybe;

      if (at::globalContext().ARCGlobal.isDebugMode())
        std::cout << "Test tid: " << i << ", size: " << liveness_size[i] << ", time: " << liveness_time[i] << ", " << delay_time[i] << ", " << real_trans_time[i] << ", csr, fp: " << liveness_csr[cur_back_num][i] << ", " << liveness_fp[cur_back_num][i] << std::endl;

      previous_i = i;


      if (remainSize < 0) {
        if (at::globalContext().ARCGlobal.isDebugMode())
          std::cout << "Remained size is zero" << std::endl;

        break;
      }
    }
  }

  bool timeout = (liveness_time[previous_i] + real_trans_time[previous_i] + delay_time[previous_i]) > last_time_slot;

  if (previous_i + 1 == NUM_TENSOR) {
    if (at::globalContext().ARCGlobal.isDebugMode())
      std::cout << "Future work: operate on-demand mode as default" << std::endl;

    exit(1);
  }

  if (timeout) {
    if (at::globalContext().ARCGlobal.isDebugMode())
      std::cout << "Second phase: previous_i - " << previous_i << std::endl;

    for (int i = previous_i + 1; i < NUM_TENSOR; i++) {
      int delete_i = 0;
      for (delete_i = previous_i; delete_i >= 0; delete_i--) {
        if (liveness_size[delete_i] > 1 && at::native::arc_vm.liveness_result[cur_back_num][delete_i] &&
            !liveness_csr[cur_back_num][delete_i] && !liveness_fp[cur_back_num][delete_i]) {
          break;
        }
      }

      if (delete_i == -1 && !at::globalContext().ARCGlobal.isBERT()) {
        for (delete_i = previous_i; delete_i >= 0; delete_i--) {
          if (liveness_size[delete_i] > 1 && at::native::arc_vm.liveness_result[cur_back_num][delete_i] && !liveness_csr[cur_back_num][delete_i]) {
            break;
          }
        }
      }

      if (at::globalContext().ARCGlobal.isDebugMode())
        std::cout << "Deleted tid: " << delete_i << ", csr, fp: " << liveness_csr[cur_back_num][delete_i] << ", " << liveness_fp[cur_back_num][delete_i] << ", size: " << liveness_size[delete_i] << std::endl;

      double delete_size = liveness_size[delete_i];

      int delete_previous_i = 0;
      if (delete_i == -1) {
        if (at::globalContext().ARCGlobal.isDebugMode())
          std::cout << "Give up in deleting" << std::endl;

        break;
      }

      at::native::arc_vm.liveness_result[cur_back_num][delete_i] = false;

      int add_i = 0;

      while (delete_size > 0) {
        for (delete_previous_i = delete_i; delete_previous_i >= 0; delete_previous_i--) {
          if (at::native::arc_vm.liveness_result[cur_back_num][delete_previous_i]) {
            break;
          }
        }

        for (add_i = previous_i + 1; add_i < NUM_TENSOR; add_i++) {
          if (liveness_size[add_i] > 1 && liveness_csr[cur_back_num][add_i] && !at::native::arc_vm.liveness_result[cur_back_num][add_i]) {
            break;
          }
        }

        if (add_i == NUM_TENSOR) {
          for (add_i = previous_i + 1; add_i < NUM_TENSOR; add_i++) {
            if (liveness_size[add_i] > 1 && liveness_fp[cur_back_num][add_i] && !at::native::arc_vm.liveness_result[cur_back_num][add_i]) {
              break;
            }
          }
        }

        if (add_i == NUM_TENSOR) {
          std::cout << "GG" << std::endl;
          break;
        }

        at::native::arc_vm.liveness_result[cur_back_num][add_i] = true;
        for (int test = delete_i; test <= add_i; test++) {
          if (at::native::arc_vm.liveness_result[cur_back_num][test]) {
            if (delete_previous_i != -1) {
              double delay_maybe = liveness_time[delete_previous_i] + real_trans_time[delete_previous_i] + delay_time[delete_previous_i] - liveness_time[test];
              if (delay_maybe <= 0) {
                delay_time[test] = 0;
              } else {
                delay_time[test] = delay_maybe;
              }
            } else {
              delay_time[test] = 0;
            }

            delete_previous_i = test;
          }
        }

        if (at::globalContext().ARCGlobal.isDebugMode())
          std::cout << "Added tid test: " << add_i << ", size: " << liveness_size[add_i] << ", time info: " << liveness_time[add_i] << ", " << delay_time[add_i] << ", " << real_trans_time[add_i] << ", " << liveness_time[add_i]+delay_time[add_i]+real_trans_time[add_i]  << ", csr/fp: " << liveness_csr[cur_back_num][add_i] << ", " << liveness_fp[cur_back_num][add_i] << ", " << last_time_slot << std::endl;

        i = add_i;
        delete_size -= liveness_size[add_i];
      }

      bool timeout = (liveness_time[add_i] + real_trans_time[add_i] + delay_time[add_i]) > last_time_slot;
      if (!timeout) {
        if (at::globalContext().ARCGlobal.isDebugMode())
          std::cout << "Delete done" << std::endl;

        break;
      }
    }

    if (remainSize > 0) {
      for (int i = 0; i < NUM_TENSOR; i++) {
        if (liveness_size[i] > 1 && !at::native::arc_vm.liveness_result[cur_back_num][i] && liveness_csr[cur_back_num][i]) {
          remainSize -= liveness_size[i];
          at::native::arc_vm.liveness_result[cur_back_num][i] = true;
        }

        if (remainSize < 0) break;
      }
    }

    if (remainSize > 0) {
      for (int i = 0; i < NUM_TENSOR; i++) {
        if (liveness_size[i] > 1 && !at::native::arc_vm.liveness_result[cur_back_num][i] && liveness_fp[cur_back_num][i]) {
          remainSize -= liveness_size[i];
          at::native::arc_vm.liveness_result[cur_back_num][i] = true;
        }

        if (remainSize < 0) break;
      }
    }
  }
  
  if (at::globalContext().ARCGlobal.isDebugMode()) {
    std::cout << "Remained data: " << remainSize << std::endl;
    std::cout << "Forward end time: " << last_time_slot << std::endl;
    std::cout << "TID, size, csr, fp, time, delay, trans time" << std::endl;

    double selected_csr = 0;
    double non_selected_csr = 0;
    double selected_fp = 0;
    double non_selected_fp = 0;
    double selected_norm = 0;
    double non_selected_norm = 0;

    for (int i = 0; i < NUM_TENSOR; i++) {
      if (at::native::arc_vm.liveness_result[cur_back_num][i]) {
        if (liveness_csr[cur_back_num][i]) selected_csr += liveness_size[i];
        else if (liveness_fp[cur_back_num][i]) selected_fp += liveness_size[i];
        else selected_norm += liveness_size[i];

        std::cout << i << ", " << liveness_size[i] << ", " << liveness_csr[cur_back_num][i] << ", " << liveness_fp[cur_back_num][i] << ", " << liveness_time[i] << ", " << delay_time[i] << ", " << real_trans_time[i] << ", " << std::endl;
      } else {
        if (liveness_csr[cur_back_num][i]) non_selected_csr += liveness_size[i];
        else if (liveness_fp[cur_back_num][i]) non_selected_fp += liveness_size[i];
        else non_selected_norm += liveness_size[i];
      }
    }

    std::cout << "Size classification" << std::endl;
    std::cout << "CSR tensor size classification: " << selected_csr << ", " << non_selected_csr << std::endl;
    std::cout << "FP tensor size classification: " << selected_fp << ", " << non_selected_fp << std::endl;
    std::cout << "norm tensor size classification: " << selected_norm << ", " << non_selected_norm << std::endl;
  }
}


//Note: Not referecne but copy a tensor to make it alive
void ARCCppEngine::offLoad(at::Tensor t, /*TraceableFunction* grad_fn, ARCSync sync,*/ Oid oid, SavedVariable* fetch_loc, bool isOutput) {

  if (!at::native::arc_vm.is_vdnn()) {
    *fetch_loc = SavedVariable(t, isOutput);
    return;
  }

  // partial offloading
  auto tid =  t.unsafeGetIntrusivePtr()->tensor_id;
  at::native::arc_vm.feature_map_accum[tid] = (double)t.nbytes() / 1024 / 1024;
  int cur_back_num = at::globalContext().ARCGlobal.curBackNum();

  if (at::native::arc_vm.liveness_result[cur_back_num][tid] == false && !at::globalContext().ARCGlobal.isOnDemand()) {
    *fetch_loc = SavedVariable(t, isOutput);
    return;
  }

  if (tid == 0) {
    *fetch_loc = SavedVariable(t, isOutput);
    return;
  }

  insertToPFDict_(oid, fetch_loc, tid);
  // save the information required for prefetching
  if (offload_sync[tid]) {// this tensor is alredy offloaded
    if (at::globalContext().ARCGlobal.isDebugMode())
      std::cout << tid <<  ": this tensor is already offloaded" << std::endl;

    at::native::arc_vm.relu_thru = false;
    return;
  }

  if (at::globalContext().ARCGlobal.isOnDemand() && at::globalContext().ARCGlobal.isForward()) {

    double elapsed = 0;
    if (accumSize > 0) {
      gettimeofday(&at::native::arc_vm.tv2, NULL);
      elapsed = (at::native::arc_vm.tv2.tv_sec - at::native::arc_vm.tv1.tv_sec) * 1000 +
          (double)(at::native::arc_vm.tv2.tv_usec - at::native::arc_vm.tv1.tv_usec) / 1000;
    }

    accumSize += (double)t.nbytes() / 1024 / 1024;

    if (at::globalContext().ARCGlobal.isDebugMode()) {
      std::cout << "accumulated tensor tid: " << tid << ", size: " << (double)t.nbytes() / 1024 / 1024 << ", accum: " << accumSize << ", relu_thru: " << at::native::arc_vm.relu_thru << std::endl;
    }

    liveness_time[tid] = elapsed;
    liveness_size[tid] = (double)t.nbytes() / 1024 / 1024;

    if (at::native::arc_vm.is_csr())
      liveness_csr[cur_back_num][tid] = at::native::arc_vm.relu_thru;
    else
      liveness_csr[cur_back_num][tid] = false;

    if (at::native::arc_vm.is_fp16()) {
      if (t.element_size() >= 4)
        liveness_fp[cur_back_num][tid] = true;
      else
        liveness_fp[cur_back_num][tid] = false;
    } else {
      liveness_fp[cur_back_num][tid] = false;
    }

    at::native::arc_vm.relu_thru = false;
  }

  offload_sync[tid] = true;

  auto str = c10::cuda::getStreamFromPool(false, 0);
//  str.synchronize();
  c10::cuda::CUDAStreamGuard csg(str);
  c10::TensorOptions opt = c10::TensorOptions();
  opt = opt.device(c10::Device(c10::DeviceType::CPU));
  opt = opt.dtype(t.dtype());
  opt = opt.pinned_memory(true);

  // [JS] p2p setting if ssd mode is on
  if (at::native::arc_vm.is_using_ssd()) {
    at::native::arc_vm.set_dir(tid, at::native::arcp2p_gputossd);
    at::native::arcp2p_cpl *p_cpl = new at::native::arcp2p_cpl;
    at::native::arc_vm.set_cpl_addr(tid, at::native::arcp2p_gputossd, (void *)p_cpl);
  }

  if (at::globalContext().ARCGlobal.isOnDemand()) {
    target_tensor[tid] = t.ARCto(opt, false, true, false);
    target_tensor_valid[tid] = true;

    while (at::native::arc_vm.event_arr_d2h[tid]) {
      if (at::native::arc_vm.is_using_ssd()) {
        at::native::arc_vm.Arcp2pCompletion(false);
      }
    }

    last_use_forward[cur_back_num][tid] = oid;   

  } else {
    if (last_use_forward[cur_back_num][tid] == oid) {
      target_tensor[tid] = t.ARCto(opt, false, true, liveness_csr[cur_back_num][tid]);
      target_tensor_valid[tid] = true;
    }
  }

  if (at::native::arc_vm.is_using_ssd())
    at::native::arc_vm.Arcp2pCompletion(false);

  csg.reset_stream(csg.original_stream());

  if (at::globalContext().ARCGlobal.isOnDemand() && at::globalContext().ARCGlobal.isForward()) {
    gettimeofday(&at::native::arc_vm.tv1, NULL);
  }
}

void ARCCppEngine::joinOffload() {
  if (at::globalContext().ARCGlobal.isDebugMode())
    std::cout << "Wait until all offloading is done" << std::endl;

  if (at::globalContext().ARCGlobal.isOnDemand()) {
    gettimeofday(&at::native::arc_vm.tv2, NULL);
    last_time_slot = (at::native::arc_vm.tv2.tv_sec - at::native::arc_vm.tv1.tv_sec) * 1000 +
          (double)(at::native::arc_vm.tv2.tv_usec - at::native::arc_vm.tv1.tv_usec) / 1000;
  }

  if (at::globalContext().ARCGlobal.isDebugMode())
    std::cout << "Wait end" << std::endl;
}

void ARCCppEngine::preFetchSync(Oid oid, bool isOutput) { 
  //this operation has nothing to prefetch 
  if (oid == 0)
    return;

  if (!at::native::arc_vm.is_vdnn()) {
    return;
  }

  if (op_tensor_list.find(oid) == op_tensor_list.end()) {
    return;
  }

  while (1) {
    auto check = stream_occupied.find(oid);
    if (check != stream_occupied.end())
      break;
  }

  auto sid = stream_occupied[oid];
  c10::cuda::CUDAStream str(c10::Stream(c10::Stream::UNSAFE, c10::Device(c10::DeviceType::CUDA, 0), sid));
  str.synchronize();
  cudaStreamSynchronize(at::native::arc_vm.arc_stream);

  auto fetch_vec = op_tensor_list[oid];
  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;
    auto fetch_loc = it->first;

    if (prefetch_sync[tid] == true) {
      *fetch_loc = SavedVariable(target_tensor[tid], isOutput);
      continue;
    }

    if (at::globalContext().ARCGlobal.isDebugMode())
      std::cout << "tid " << tid << " in oid " << oid << " pf sync start" << std::endl;

    if (target_tensor_valid[tid] == false) {
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

          prefetch_sync[tid] = true;
          break;
        }
        at::native::arc_vm.Arcp2pCompletion(true);
      } else {
        if (target_tensor[tid].device().type() == c10::DeviceType::CUDA) {
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

          prefetch_sync[tid] = true;
          at::native::arc_vm.event_arr_h2d[tid] = false;
          break;
        }
      }
    }

    *fetch_loc = SavedVariable(target_tensor[tid], isOutput);
  }
}

void ARCCppEngine::insertToPFDict_(Oid oid, SavedVariable* loc, Tid tid) {
  //lookup and insert op_tensor_list
  auto it = op_tensor_list.find(oid);
  if (it != op_tensor_list.end()) {
    op_tensor_list[oid].emplace_back(loc, tid);
  } else {
    std::vector<PFInfo> tmp;
    tmp.emplace_back(loc, tid);
    op_tensor_list.insert(std::pair<Oid, std::vector<PFInfo>>(oid, tmp));
  }
}

// drop tensor from gpu memory
// this function should be called at the end of back prop of each layer
void ARCCppEngine::dropTensor(Oid oid, SavedVariable* fetch_loc) { 
  //this operation has nothing to prefetch 
  if (op_tensor_list.find(oid) == op_tensor_list.end()) {
    //std::cerr << oid << " Prefetching dictionary lookup miss" << std::endl;
    return;
  }

  auto fetch_vec = op_tensor_list[oid];
  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;
    if (at::globalContext().ARCGlobal.isOnDemand()) {
      at::Tensor& tref = target_tensor[tid]; 
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
      prefetch_sync[tid] = false;

      while (at::native::arc_vm.event_arr_d2h[tid]) {
        if (at::native::arc_vm.is_using_ssd()) {
          at::native::arc_vm.Arcp2pCompletion(false);
        }
      }
    } else {
      int cur_back_num = at::globalContext().ARCGlobal.curBackNum();
      if ((oid == last_use_backward[cur_back_num][tid]) && target_tensor_valid[tid]) {
//        std::cout << "dropTensor: " << tid << std::endl;
        target_tensor_valid[tid] = false;
        fetch_loc->reset_data();
        c10::cuda::CUDACachingAllocator::emptyCache();

//        if (at::native::arc_vm.hard_training)
//          at::native::arc_vm.Arcp2pCompletion(true);
      }
    }
  }
}

bool ARCCppEngine::preFetch(Oid oid) {
  if (oid == 0)
    return false;

  if (!at::native::arc_vm.is_vdnn()) {
    return false;
  }

  //this operation has nothing to prefetch 
  if (op_tensor_list.find(oid) == op_tensor_list.end()) {
    //std::cerr << oid << " Prefetching dictionary lookup miss" << std::endl;
    return true;
  }

  if (at::globalContext().ARCGlobal.isOnDemand()) {
    at::globalContext().ARCGlobal.pushBackOid(oid);
  }
  
  auto fetch_vec = op_tensor_list[oid];
  int cur_back_num = at::globalContext().ARCGlobal.curBackNum();

  auto str = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard csg(str);
  stream_occupied.insert(std::pair<Oid, c10::StreamId>(oid, str.id()));

  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;

    if (target_tensor_valid[tid] == false) {
      //std::cerr << "tensor dictionary lookup miss" << std::endl;
      return true;
    }

    at::Tensor& tref = target_tensor[tid];
    c10::TensorOptions opt = c10::TensorOptions();
    opt = opt.device(c10::Device(c10::DeviceType::CUDA));
    opt = opt.dtype(tref.dtype());
    //opt = opt.dtype(c10::ScalarType::Float);
 
    if (tref.device().type() == c10::DeviceType::CPU) {

      if (!at::globalContext().ARCGlobal.isOnDemand()) {
//        if (at::native::arc_vm.device_occupancy_future(tref.nbytes()) < 0.05) {
//          c10::cuda::CUDACachingAllocator::emptyCache();
//          return false;
//        }

        if (at::native::arc_vm.on_the_fly > 1) {
          c10::cuda::CUDACachingAllocator::emptyCache();
          return false;
        }
      }

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
        last_use_backward[cur_back_num][tid] = oid;   
      }

      if (at::globalContext().ARCGlobal.isOnDemand()) {
        tref = tref.ARCto(opt, false, true, false);

        while (at::native::arc_vm.event_arr_h2d[tid]) {
          at::native::arc_vm.Arcp2pCompletion(false);
        }
      } else {
        tref = tref.ARCto(opt, false, true, liveness_csr[cur_back_num][tid]);
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
        remaining_backward = backward_num_BERT;
    }
  }
  

  for(auto i = 0; i < NUM_TENSOR; i ++) {
    if (prefetch_sync[i] == true) {
      target_tensor[i].reset();
    }
  }

  memset(target_tensor_valid, 0, sizeof(bool) * NUM_TENSOR);
  memset(offload_sync, 0, sizeof(bool) * NUM_TENSOR);
  memset(prefetch_sync, 0, sizeof(bool) * NUM_TENSOR);

  memset(liveness_time, 0, sizeof(double) * NUM_TENSOR);
  memset(liveness_size, 0, sizeof(double) * NUM_TENSOR);
//  memset(liveness_csr, 0, sizeof(bool) * NUM_TENSOR);
//  memset(liveness_fp, 0, sizeof(bool) * NUM_TENSOR);

  op_tensor_list.clear();
  stream_occupied.clear();

  --remaining_backward;
  if (remaining_backward == 0) {
    at::globalContext().ARCGlobal.resetGlobalTid();
    at::globalContext().ARCGlobal.resetGlobalOid();

    double accum_sum = 0;
    for(int i = 0; i < NUM_TENSOR; i++) {
      if (at::native::arc_vm.feature_map_accum[i] > 0) {
//        std::cout << "accum tid size: " << i << ", " << at::native::arc_vm.feature_map_accum[i] << std::endl;
        accum_sum += at::native::arc_vm.feature_map_accum[i];
      }

      at::native::arc_vm.feature_map_accum[i] = 0;
    }

/*
    std::cout << "Accumulated feature map: " << accum_sum << " MB" << std::endl;
    std::cout << "Accumulated gradient map: " << at::native::arc_vm.gradient_map_accum << " MB" << std::endl;
    std::cout << "Accumulated weight: " << at::native::arc_vm.weight_accum << " MB" << std::endl;
    std::cout << "Accumulated missing?: " << at::native::arc_vm.misc_accum << " MB" << std::endl;
*/

    at::native::arc_vm.gradient_map_accum = 0;
    at::native::arc_vm.weight_accum = 0;
    at::native::arc_vm.misc_accum = 0;

    if (at::globalContext().ARCGlobal.isCycleGAN()) {
        remaining_backward = backward_num_CycleGAN;
    } else {
//    if (at::globalContext().ARCGlobal.isBERT())
        remaining_backward = backward_num_BERT;
    }
  }
}

}} // namespace torch::autograd
