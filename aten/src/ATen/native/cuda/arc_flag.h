#pragma once

#include <ATen/ATen.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>
#include <stdint.h>
#include <stdbool.h>

// [JS] for p2p library
#include <c10/core/Storage.h>
#include <dlfcn.h>
#include <c10/cuda/CUDAStream.h>

using namespace std;
namespace at { namespace native {

typedef struct
{
  uint64_t tid;
  uint64_t numel;
  uint64_t ntpb;
  void *dst;
  void *src;
  void *ptr;
} arcp2p_info;

#include "arcp2p.h"
using arcp2p_type1_fn = arcp2p * (*)(const char *[PATH_LENGTH], int);
using arcp2p_type2_fn = int      (*)(arcp2p *);
using arcp2p_type3_fn = int      (*)(arcp2p *, uint64_t, uint64_t);
using arcp2p_type4_fn = int      (*)(arcp2p *, uint64_t, uint64_t, uint64_t, arcp2p_cpl *, arcp2p_dir);

class ARC_memory {
public:
  ARC_memory();
  ~ARC_memory();

  bool relu_thru;

  void* get_bit_addr(int tid);
  void set_bit_addr(int tid, uint64_t addr);

  void* get_pos_addr(int tid);
  void set_pos_addr(int tid, uint64_t addr);

  unsigned int get_resize(int tid);
  void set_resize(int tid, unsigned int resize);

  void* get_cpl_addr(int tid, arcp2p_dir dir);
  void set_cpl_addr(int tid, arcp2p_dir dir, void *addr);

  uint64_t* get_offset_ptr(int tid);

  arcp2p_dir get_dir(int tid);
  void set_dir(int tid, arcp2p_dir dir);

  // [JS] P2P library
  void Arcp2pSetting(int flags);
  int  Arcp2pBarMapping(uint64_t, uint64_t);
  void Arcp2pSubmission(uint64_t, uint64_t, uint64_t *, arcp2p_cpl *, arcp2p_dir, c10::Storage *, arcp2p_info *);
  void Arcp2pCompletion();

  // [JS] flag check
  bool is_using_ssd();
  bool is_fp16();
  bool is_csr();

private:
  uint64_t* bit_ptr_arr;
  uint64_t* pos_ptr_arr;
  unsigned int* resize_arr;
  uint64_t* cpl_flu_ptr_arr;
  uint64_t* cpl_pre_ptr_arr;
  uint64_t* offset_arr;
  arcp2p_dir* dir_arr;

  // [JS] P2P library
  arcp2p_type1_fn arcp2p_initialize;
  arcp2p_type2_fn arcp2p_release;
  arcp2p_type3_fn arcp2p_bar_attach;
  arcp2p_type2_fn arcp2p_bar_detach;
  arcp2p_type4_fn arcp2p_transfer;
  arcp2p_type2_fn arcp2p_completion;

  arcp2p *arc_handle;
  uint64_t last_allocated_offset;
  bool isTesla;
  bool isUsingSSD;
  bool isFP16;
  bool isCSR;
};

extern ARC_memory arc_vm;

}} // namespace at::native
