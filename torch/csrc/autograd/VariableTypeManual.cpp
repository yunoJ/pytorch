#include <c10/util/Optional.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/utils/memory.h>

#include <torch/csrc/utils/memory.h>

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

namespace {
std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(at::ArrayRef<at::Backend> backends) {
  std::vector<DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for (int64_t s = 0; s < static_cast<int64_t>(ScalarType::NumOptions); s++) {
      auto& type = getNonVariableDeprecatedTypeProperties(static_cast<Backend>(p), static_cast<ScalarType>(s));
      res.emplace_back(&type);
    }
  }
  return res;
}
}

std::vector<at::DeprecatedTypeProperties*> VariableType::allCPUTypes() {
  return allTypesForBackends({ Backend::CPU, Backend::SparseCPU });
}

std::vector<at::DeprecatedTypeProperties*> VariableType::allCUDATypes() {
  at::globalContext().lazyInitCUDA();
  return allTypesForBackends({ Backend::CUDA, Backend::SparseCUDA });
}

const Variable & VariableType::checked_cast_variable(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  if (!t.is_variable()) {
    AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " for argument #", pos, " '", name, "'");
  }
  return as_variable_ref(t);
}

Variable & VariableType::checked_cast_variable(Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  if (!t.is_variable()) {
    AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " for argument #", pos, " '", name, "'");
  }
  return as_variable_ref(t);
}

const Tensor & VariableType::unpack(const Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor & VariableType::unpack(Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor VariableType::unpack_opt(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

std::vector<at::Tensor> VariableType::unpack(at::TensorList tl, const char *name, int pos) {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      continue;
    }
    if (!t.is_variable()) {
      AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " at position #", i, " "
                    "for iterable argument #", pos, " '", name, "'");
    }
    ret[i] = static_cast<const Variable&>(t);
  }
  return ret;
}

void VariableType::backward(
    const Tensor& self,
    const Tensor& gradient,
    bool keep_graph,
    bool create_graph) {
  as_variable_ref(self).backward(gradient, keep_graph, create_graph);
}

void VariableType::set_data(const Tensor & self, const Tensor & new_data) {
  as_variable_ref(self).set_data(new_data);
}

// We don't have an outplace copy, so this can't be generated automatically
Tensor & VariableType::copy_(Tensor & self, const Tensor & src, bool non_blocking) {
  jit::Value* output = nullptr;
  if(torch::jit::tracer::isTracing()) {
    const jit::tracer::TracingState& state = *jit::tracer::getTracingState();
    auto& graph = state.graph;
    if (state.force_outplace) {
      // if you have no views of self, then an in place copy is equivalent to
      // making sure we expand src to the same size as self
      jit::Node* node = graph->create(jit::aten::expand_as, /*num_outputs=*/1);
      jit::tracer::addInputs(node, "src", src);
      jit::tracer::addInputs(node, "self", self);
      graph->insertNode(node);
      jit::tracer::ensureUniqueIfOutOfPlaced("copy_ (possibly due to an assignment)", self);
      output = node->output();
    } else {
      output = graph->insert(
          jit::aten::copy_,
          {jit::tracer::getValueTrace(self), jit::tracer::getValueTrace(src)});
    }
  }
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isFloatingPoint(self.scalar_type());
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->set_next_edges(collect_next_edges(self, src));
    grad_fn->src_type = &src.type();
    grad_fn->src_device = src.device();
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.copy_(src_, non_blocking);
  }
  increment_version(self);
  rebase_history(as_variable_ref( self ), std::move(grad_fn));
  if(torch::jit::tracer::isTracing()) {
    jit::tracer::setOutput(output, self);
  }
  return self;
}

Tensor & VariableType::ARCcopy_(Tensor & self, const Tensor & src, bool non_blocking, bool is_csr) {
  /*
  jit::Value* output = nullptr;
  if(torch::jit::tracer::isTracing()) {
    const jit::tracer::TracingState& state = *jit::tracer::getTracingState();
    auto& graph = state.graph;
    if (state.force_outplace) {
      // if you have no views of self, then an in place copy is equivalent to
      // making sure we expand src to the same size as self
      jit::Node* node = graph->create(jit::aten::expand_as, 1);
      jit::tracer::addInputs(node, "src", src);
      jit::tracer::addInputs(node, "self", self);
      graph->insertNode(node);
      jit::tracer::ensureUniqueIfOutOfPlaced("ARCcopy_ (possibly due to an assignment)", self);
      output = node->output();
    } else {
      output = graph->insert(
          jit::aten::ARCcopy_,
          {jit::tracer::getValueTrace(self), jit::tracer::getValueTrace(src)});
    }
  }
  */
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  /*
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isFloatingPoint(self.scalar_type());
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->set_next_edges(collect_next_edges(self, src));
    grad_fn->src_type = &src.type();
    grad_fn->src_device = src.device();
  }
  */
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.ARCcopy_(src_, non_blocking, is_csr);
  }
  increment_version(self);
  /*
  rebase_history(as_variable_ref( self ), std::move(grad_fn));
  if(torch::jit::tracer::isTracing()) {
    jit::tracer::setOutput(output, self);
  }
  */
  return self;
}

Tensor & VariableType::resize_(Tensor & self, IntArrayRef size) {
  auto& self_ = unpack(self, "self", 0);
  if (as_variable_ref(self).requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  if (torch::jit::tracer::isTracing()) {
    jit::tracer::ArgumentStash::popIntArrayRef("size");
    jit::tracer::warn("resize_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.resize_(size);
  }
  return self;
}

Tensor & VariableType::resize_as_(Tensor & self, const Tensor & the_template) {
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  if (as_variable_ref(self).requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  if (torch::jit::tracer::isTracing()) {
    jit::tracer::warn("resize_as_", jit::tracer::WARN_RESIZE);
    jit::tracer::delValueTrace(self);
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::resize_as_(self_, the_template_);
  }
  return self;
}

Tensor VariableType::detach(const Tensor & self) {
  RECORD_FUNCTION("detach", std::vector<c10::IValue>({self}));

  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);

  }
  // <NON_GENERATED_CODE>
  auto result = as_variable_ref(const_cast<Tensor&>(self)).detach(); // NOLINT(cppcoreguidelines-pro-type-const-cast)
  // </NON_GENERATED_CODE>
  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, result);
  }
  return std::move(result);
}

Tensor & VariableType::detach_(Tensor & self) {
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self}));

  torch::jit::Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    auto& graph = jit::tracer::getTracingState()->graph;
    node = graph->create(jit::aten::detach, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("detach_", self);
  }
  // <NON_GENERATED_CODE>
  as_variable_ref(self).detach_();
  // </NON_GENERATED_CODE>
  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, self);
  }
  return self;
}

}} // namespace torch::autograd
