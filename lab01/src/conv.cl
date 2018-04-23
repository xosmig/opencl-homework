__constant size_t MAX_COEF_SIZE = 10;

__kernel void conv(__global float* m, size_t m_size,
                         __global float* coef_glob, size_t coef_size,
                         __global float* res) {
  size_t id_x = get_global_id(0);
  size_t id_y = get_global_id(1);

  __local float coef[MAX_COEF_SIZE][MAX_COEF_SIZE];

  size_t loc_x = get_local_id(0);
  size_t loc_y = get_local_id(1);
  if (loc_x < coef_size && loc_y < coef_size) {
    coef[loc_x][loc_y] = coef_glob[loc_x * coef_size + loc_y];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  bool allow_write = id_x < m_size && id_y < m_size;
  if (!allow_write) {
    return;
  }

  size_t res_pos = id_x * m_size + id_y;
  res[res_pos] = 0;
  for (long long dx = 0; dx < coef_size; ++dx) {
    for (long long dy = 0; dy < coef_size; ++dy) {
      long long x = id_x + dx - coef_size / 2;
      long long y = id_y + dy - coef_size / 2;
      if (x < 0 || y < 0 || x >= m_size || y >= m_size) {
        continue;
      }
      res[res_pos] += m[x * m_size + y] * coef[dx][dy];
      // res[res_pos] += m[x * m_size + y] * coef_glob[dx * coef_size + dy];
    }
  }
}
