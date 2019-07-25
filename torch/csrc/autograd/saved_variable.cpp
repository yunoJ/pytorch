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
static std::map<Oid,std::vector<bool>> pf_sync_dict_;
static std::map<Tid,at::Tensor> tensor_dict_;


//Note: Not referecne but copy a tensor to make it alive
void ARCCppEngine::offLoad(at::Tensor& t, TraceableFunction* grad_fn, ARCSync sync, Oid curOid, SavedVariable* fetch_loc) {
  grad_fn->incrTNum();
  
  if (sync == Async) memcpy_threads_.push_back(std::thread(dtoh_, t, curOid, fetch_loc));
 
  //dumbest sync policy. Need optimized.
  if(sync == Sync) {
    std::thread worker(dtoh_, t, curOid, fetch_loc);
    worker.join();
  }
}

//Note: Not referecne but copy a tensor to make it alive
void ARCCppEngine::fetch_(at::Tensor& t, Oid oid, ARCSync sync, SavedVariable* fetch_loc) { 
  if (sync == Async)memcpy_threads_.push_back(std::thread(htod_, t, oid, fetch_loc));
  
  //dumbest sync policy. Need optimized.
  if(sync == Sync) {
    std::thread worker(htod_, t, oid, fetch_loc);
    worker.join();
    //explicitAllSync();
  }
}

void ARCCppEngine::explicitAllSync() {
  for (auto it = memcpy_threads_.begin(); it != memcpy_threads_.end(); ++it)
      it->join();
  memcpy_threads_.clear();
}

void ARCCppEngine::preFetch(Oid curOid, int required_tensor_num, ARCSync sync) {
  Oid target = whoWillPrefetched_(curOid);
  if (target < 0) std::cerr << "There is no more operation to need prefetch." << std::endl;
  fetchRequiredTensors_(target, required_tensor_num, sync); 
}

void preFetchSync(Oid oid, int required_tensor_num) {
  while (1) {
    auto check = pf_sync_dict_.find(oid);
    if (check != pf_sync_dict_.end()) {
      auto vec = pf_sync_dict_[oid];
      if (vec.size() == required_tensor_num)
        break;
    }
  }
}

//for fetching/offloaidng
void ARCCppEngine::htod_(at::Tensor t, Oid oid, SavedVariable* fetch_loc) {
  c10::TensorOptions opt = c10::TensorOptions();
  opt = opt.device(c10::Device(c10::DeviceType::CUDA));
  opt = opt.dtype(t.scalar_type());
  int stored_id = t.unsafeGetTensorImpl()->tensor_id;


  at::Tensor device_t = t.to(opt);
  device_t.unsafeGetTensorImpl()->tensor_id = stored_id;

  if (fetch_loc) {
    *fetch_loc = SavedVariable(device_t, false); // need to support true case?
    insertToPFSyncDict_(oid); 
  } else {
    //Tensor device_t = t.to(Device(DeviceType::CUDA), t.scalar_type());
    t.unsafeGetIntrusivePtr().swap(device_t.unsafeGetIntrusivePtr());
    t.unsafeGetTensorImpl()->tensor_id = stored_id;
    device_t.reset();
  }
}

// fetch_loc may be null in on-demand mode
// and will be not null in non-on-demand mode
void ARCCppEngine::dtoh_(at::Tensor t, Oid oid, SavedVariable* fetch_loc) {
  c10::TensorOptions opt = c10::TensorOptions();
  opt = opt.device(c10::Device(c10::DeviceType::CPU));
  opt = opt.dtype(t.scalar_type());
  int stored_id = t.unsafeGetTensorImpl()->tensor_id; 
  
  at::Tensor host_t = t.to(opt);
  host_t.unsafeGetTensorImpl()->tensor_id = stored_id;
  if (fetch_loc) {
    insertToPFDict_(oid, fetch_loc, host_t);
    //t.unsafeGetIntrusivePtr().swap(host_t.unsafeGetIntrusivePtr());
    //t.unsafeGetTensorImpl()->tensor_id = stored_id;
    //host_t.reset();
  } else { 
    t.unsafeGetIntrusivePtr().swap(host_t.unsafeGetIntrusivePtr());
    t.unsafeGetTensorImpl()->tensor_id = stored_id;
    host_t.reset();
  }
}

void ARCCppEngine::insertToPFSyncDict_(Oid oid) {
  auto it = pf_sync_dict_.find(oid);
  if (it == pf_sync_dict_.end()) {
    std::vector<bool> tmp(1);
    pf_sync_dict_.insert(std::pair<Oid, std::vector<bool>>(oid, tmp));
  } else {
    pf_sync_dict_[oid].push_back(0);
  }
}



void ARCCppEngine::insertToTensorDict_(at::Tensor& backup) {
  auto tid = at::globalContext().ARCGlobal.getTid(backup);
  auto it = tensor_dict_.find(tid);
  if (it == tensor_dict_.end()) {
    tensor_dict_.insert(std::pair<Tid, at::Tensor>(tid, backup));
  }
}



void ARCCppEngine::insertToPFDict_(Oid oid, SavedVariable* loc, at::Tensor& backup) {
  //first lookup and insert tensor_dict_
  auto tid = at::globalContext().ARCGlobal.getTid(backup);
  insertToTensorDict_(backup);

  //then lookup and insert pf_dict_
  auto it = pf_dict_.find(oid);
  if (it != pf_dict_.end()) {
    pf_dict_[oid].emplace_back(loc, tid);
  } else {
    std::vector<PFInfo> tmp;
    tmp.emplace_back(loc, tid);
    pf_dict_.insert(std::pair<Oid, std::vector<PFInfo>>(oid, tmp));
  }
}

void ARCCppEngine::offLoadSync_(Oid oid, int required_tensor_num) { // Are all tensors required for oid's back prop offloaded?  
  // 1. pf_dict_ has all tids?
  // busy waiting
  while (1) {
    if (pf_dict_.find(oid) != pf_dict_.end())
      if (pf_dict_[oid].size() == required_tensor_num) 
        break;
  }
  auto ovec = pf_dict_[oid];

  // 2. each tid is offloaded in tensor_dict_?
  while (1) {
    bool ok = 1;
    for (auto it = ovec.begin(); it != ovec.end(); it++) {
      auto tid = it->second;
    
      if (tensor_dict_.find(tid) == tensor_dict_.end()) {
        ok = 0;
        break;
      }
    }
    if (ok) break; 
  }
}


void ARCCppEngine::fetchRequiredTensors_(Oid oid, int required_tensor_num, ARCSync sync) {
  
  // check required tensors are properly offloaded.
  offLoadSync_(oid, required_tensor_num);
  
  if (pf_dict_.find(oid) == pf_dict_.end())
    std::cerr << "Prefetching dictionary lookup miss" << std::endl;

  auto fetch_vec = pf_dict_[oid]; 
  for (auto it = fetch_vec.begin(); it != fetch_vec.end(); it++) {
    auto tid = it->second;
    auto fetch_loc = it->first;

    if (tensor_dict_.find(tid) == tensor_dict_.end())
      std::cerr << "Prefetching dictionary lookup miss" << std::endl;
    auto backup = tensor_dict_[tid];
    fetch_(backup, oid, sync, fetch_loc); 
  }
}

Oid ARCCppEngine::whoWillPrefetched_(Oid curOid) {
  return curOid; //ad-hoc policy
}

void ARCCppEngine::resetCppEngine() {
  at::globalContext().ARCGlobal.resetGlobalTid();
  at::globalContext().ARCGlobal.resetGlobalOid();
  explicitAllSync();
  pf_dict_.clear();
}





}} // namespace torch::autograd
