#pragma once

#include <torch/csrc/python_headers.h>
#include <memory>
#include <ATen/ATen.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/THP_export.h>

// Python object that backs torch.autograd.Variable
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPVariable {
    PyObject_HEAD
    // Payload
    torch::autograd::Variable cdata;
    // Hooks to be run on backwards pass (corresponds to Python attr
    // '_backwards_hooks', set by 'register_hook')
    PyObject* backward_hooks = nullptr;
};

THP_API PyObject *THPVariableClass;

bool THPVariable_initModule(PyObject *module);
THP_API PyObject * THPVariable_Wrap(torch::autograd::Variable var);

static inline bool THPVariable_CheckExact(PyObject *obj) {
  return Py_TYPE(obj) == (PyTypeObject*)THPVariableClass;
}

inline bool THPVariable_Check(PyObject *obj)
{
  return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}

inline torch::autograd::Variable& THPVariable_Unpack(PyObject* obj) {
  auto var = (THPVariable*)obj;
  return var->cdata;
}

// Implemented by SNU-ARC Function/Data Structures
// typedef

// ARCPyEngine for libtorch_python.so
// support sync fetching, offloading operation 
struct ARCPyEngine{
public:
  // basic fetch/offload operation
  static void fetch(at::Tensor &t);
  static void offLoad(at::Tensor &t);
};

//ARCPyEngine global_arc;
//ARCPyEngine& ARCPyEng();
//global arc
 

// end ARC
