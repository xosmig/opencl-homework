#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma ide diagnostic ignored "CannotResolve"


inline size_t get_idx(size_t gid, size_t period) {
  // period = 1  ---> 0, 1, 2,  3, ...
  // period = 4  ---> 3, 7, 11, 15, ...
  // period = 16 ---> 15, 31, 47, ...
  return (gid + 1) * period - 1;
}

__kernel void scan_hillis_steele_calculate(__global float* data, size_t size, size_t array_period,
                                           __local float* a, __local float* b) {
  size_t lid = get_local_id(0);
  size_t block_size = get_local_size(0);

  size_t idx = get_idx(get_global_id(0), array_period);
  bool allow_write = idx < size;

  a[lid] = b[lid] = data[idx];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (size_t s = 1; s < block_size; s <<= 1) {
    if (lid > s - 1) {
      b[lid] = a[lid] + a[lid - s];
    } else {
      b[lid] = a[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __local float* tmp = a;
    a = b;
    b = tmp;
  }

  // this barrier is not necessary, but it's better not to rely on
  // the implementation of the for loop above
  barrier(CLK_LOCAL_MEM_FENCE);
  if (allow_write) {
    data[idx] = a[lid];
  }
}

__kernel void scan_hillis_steele_propagate(__global float* data, size_t size, size_t array_period) {
  size_t lid = get_local_id(0);
  size_t block_size = get_local_size(0);

  size_t idx = get_idx(get_global_id(0), array_period);
  size_t block_id = get_group_id(0);

  if (block_id > 0) {
    __local float to_add;
    if (lid == 0) {
      // the last element of the previous block
      to_add = data[get_idx(block_id * block_size - 1, array_period)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < size && lid != block_size - 1) {
      data[idx] += to_add;
    }
  }
}

#pragma clang diagnostic pop
