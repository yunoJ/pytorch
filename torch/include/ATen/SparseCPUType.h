#pragma once

// @generated by aten/src/ATen/gen.py

#include <ATen/Context.h>
#include <ATen/Utils.h>



namespace at {

struct SparseCPUType final {
  static Tensor add(const Tensor & self, const Tensor & other, Scalar alpha);
  static Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha);
  static Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha);
  static Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format);
  static Tensor & log1p_(Tensor & self);
  static Tensor & log1p_out(Tensor & out, const Tensor & self);
  static Tensor mm(const Tensor & self, const Tensor & mat2);
  static Tensor & mm_out(Tensor & out, const Tensor & self, const Tensor & mat2);
  static Tensor mul(const Tensor & self, const Tensor & other);
  static Tensor & mul_(Tensor & self, const Tensor & other);
  static Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other);
  static Tensor narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length);
  static Tensor & _sparse_add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha);
  static Tensor & _sparse_div_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other);
  static Tensor & _sparse_div_scalar_out(Tensor & out, const Tensor & self, Scalar other);
  static Tensor & _sparse_mul_out(Tensor & out, const Tensor & self, const Tensor & other);
  static Tensor & _sparse_mul_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other);
  static Tensor & _sparse_mul_scalar_out(Tensor & out, const Tensor & self, Scalar other);
  static Tensor & sspaddmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha);
  static Tensor native_norm(const Tensor & self, Scalar p);
  static Tensor _sparse_sum_backward(const Tensor & grad, const Tensor & self, IntArrayRef dim);
  static Tensor clone(const Tensor & self);
  static Tensor & resize_as_(Tensor & self, const Tensor & the_template);
  static Tensor & pow_out(Tensor & out, const Tensor & self, Scalar exponent);
  static Tensor pow(const Tensor & self, Scalar exponent);
  static Tensor & zero_(Tensor & self);
  static Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options);
  static Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options);
  static Tensor & sparse_resize_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim);
  static Tensor & sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim);
  static Tensor to_dense(const Tensor & self);
  static int64_t sparse_dim(const Tensor & self);
  static int64_t dense_dim(const Tensor & self);
  static int64_t _nnz(const Tensor & self);
  static Tensor coalesce(const Tensor & self);
  static bool is_coalesced(const Tensor & self);
  static Tensor _indices(const Tensor & self);
  static Tensor _values(const Tensor & self);
  static Tensor & _coalesced_(Tensor & self, bool coalesced);
  static Tensor indices(const Tensor & self);
  static Tensor values(const Tensor & self);
  static Tensor & hspmm_out(Tensor & out, const Tensor & mat1, const Tensor & mat2);
  static Tensor hspmm(const Tensor & mat1, const Tensor & mat2);
  static Tensor & copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking);
};

} // namespace at
