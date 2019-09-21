#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Optional.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <ATen/native/cuda/arc_flag.h>

namespace at {
namespace native {

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl = c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline Tensor to_impl(const Tensor& self, const TensorOptions& options, bool non_blocking) {
  Tensor r;
  //if (options.device().type() == c10::DeviceType::CPU) {
  //  TensorOptions new_options = options;
  //  new_options = new_options.pinned_memory(true);
  //  r = at::empty(self.sizes(), new_options);
  //} else {
  auto tid = self.getIntrusivePtr().get()->tensor_id; 
  r = at::empty(self.sizes(), options);
  //}
  r.copy_(self, non_blocking);
  r.unsafeGetTensorImpl()->tensor_id = tid;
  return r;
}

static inline Tensor ARCto_impl(const Tensor& self, const TensorOptions& options, bool non_blocking, bool is_csr) {
  Tensor r;
  //if (options.device().type() == c10::DeviceType::CPU) {
  //  TensorOptions new_options = options;
  //  new_options = new_options.pinned_memory(true);
  //  r = at::empty(self.sizes(), new_options);
  //} else {
  auto tid = self.getIntrusivePtr().get()->tensor_id;

//  std::cout << "device: " << options.device().type() << ", off: " << at::native::arc_vm.liveness_result[0][tid] << ", demand: " << at::globalContext().ARCGlobal.isOnDemand() << std::endl;
  if (options.device().type() == c10::DeviceType::CPU) {
    r = at::empty(self.sizes(), options);
  } else {
    if (at::native::arc_vm.liveness_result[0][tid]) {
      r = at::ARCempty(self.sizes(), options);
    } else {
      r = at::empty(self.sizes(), options);
    }
  }

  r.ARCcopy_(self, non_blocking, is_csr);
  r.unsafeGetTensorImpl()->tensor_id = tid;
  return r;
}

Tensor to(const Tensor& self, const TensorOptions& options, bool non_blocking, bool copy) {
  TORCH_CHECK(options.requires_grad_opt() == c10::nullopt,
           "to(options) expects unset requires_grad flag, but got "
           "options.requires_grad set as ", options.requires_grad());

  const auto & layout_opt = options.layout_opt();
  TORCH_CHECK(!layout_opt || self.layout() == layout_opt.value(),
           "to(options) doesn't support converting to a different layout, "
           "but got self.layout being ", self.layout(),
           " and options.layout set as ", options.layout());

  auto device_opt = options.device_opt();
  if (device_opt) {
    device_opt = ensure_has_index(device_opt.value());
  }
  const auto & dtype_opt = options.dtype_opt();
  if ((!device_opt || self.device() == device_opt.value()) &&
      (!dtype_opt  || self.dtype()  ==  dtype_opt.value()) && !copy) {
    return self;
  }
  auto specified_options = self.options();
  if (device_opt) {
    specified_options = specified_options.device(device_opt.value());
  }
  if (dtype_opt) {
    specified_options = specified_options.dtype(dtype_opt.value());
  }
  if (options.has_pinned_memory())
    specified_options = specified_options.pinned_memory(options.pinned_memory());
  return to_impl(self, specified_options, non_blocking);
}

Tensor ARCto(const Tensor& self, const TensorOptions& options, bool non_blocking, bool copy, bool is_csr) {
  TORCH_CHECK(options.requires_grad_opt() == c10::nullopt,
      "ARCto(options) expects unset requires_grad flag, but got "
      "options.requires_grad set as ", options.requires_grad());

  const auto & layout_opt = options.layout_opt();
  TORCH_CHECK(!layout_opt || self.layout() == layout_opt.value(),
      "ARCto(options) doesn't support converting to a different layout, "
      "but got self.layout being ", self.layout(),
      " and options.layout set as ", options.layout());

  auto device_opt = options.device_opt();
  if (device_opt) {
    device_opt = ensure_has_index(device_opt.value());
  }
  const auto & dtype_opt = options.dtype_opt();
  if ((!device_opt || self.device() == device_opt.value()) &&
      (!dtype_opt  || self.dtype()  ==  dtype_opt.value()) && !copy) {
    return self;
  }
  auto specified_options = self.options();
  if (device_opt) {
    specified_options = specified_options.device(device_opt.value());
  }
  if (dtype_opt) {
    specified_options = specified_options.dtype(dtype_opt.value());
  }
  if (options.has_pinned_memory())
    specified_options = specified_options.pinned_memory(options.pinned_memory());

  return ARCto_impl(self, specified_options, non_blocking, is_csr);
}

Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy) {
  device = ensure_has_index(device);
  if (self.device() == device && self.dtype() == dtype && !copy) {
    return self;
  }
  return to_impl(self, self.options().device(device).dtype(dtype), non_blocking);
}

Tensor ARCto(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy, bool is_csr) {
  device = ensure_has_index(device);
  if (self.device() == device && self.dtype() == dtype && !copy) {
    return self;
  }
  return ARCto_impl(self, self.options().device(device).dtype(dtype), non_blocking, is_csr);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy) {
  if (self.dtype() == dtype && !copy) {
    return self;
  }
  return to_impl(self, self.options().dtype(dtype), non_blocking);
}

Tensor ARCto(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, bool is_csr) {
  if (self.dtype() == dtype && !copy) {
    return self;
  }
  return ARCto_impl(self, self.options().dtype(dtype), non_blocking, is_csr);
}

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy) {
  auto self_options = self.options();
  auto options = other.options();
  // Tensor.options() always have everything filled so we are happy and don't
  // even need to fill in device index.
  if (self_options == options && !copy) {
    return self;
  }
  return to_impl(self, options, non_blocking);
}

Tensor ARCto(const Tensor& self, const Tensor& other, bool non_blocking, bool copy, bool is_csr) {
  auto self_options = self.options();
  auto options = other.options();
  // Tensor.options() always have everything filled so we are happy and don't
  // even need to fill in device index.
  if (self_options == options && !copy) {
    return self;
  }
  return ARCto_impl(self, options, non_blocking, is_csr);
}

Tensor to_dense_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() != c10::kStrided);
  if (input_.layout() == c10::kSparse) {
    auto input = input_.coalesce();
    return grad.sparse_mask(input);
  } else if (input_.layout() == c10::kMkldnn) {
    return grad.to_mkldnn();
  } else {
    AT_ERROR("Unsupported input layout: ", input_.layout());
  }
}

Tensor to_mkldnn_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() == c10::kStrided);
  return grad.to_dense();
}

}} // namespace at::native
