#ifndef OPENCLLAB01_CL_ENVIRONMENT_HPP
#define OPENCLLAB01_CL_ENVIRONMENT_HPP

#define __CL_ENABLE_EXCEPTIONS

#include <string>
#include <iostream>
#include <CL/cl.hpp>


struct ClEnvironment {
  ClEnvironment();

  cl::Program compile_program(const std::string& file_name, std::ostream& log);

  void run_kernel_1d(const cl::Kernel& kernel, size_t global_size, size_t block_size);

  template <typename T>
  cl::Buffer create_buf_from_vector(const std::vector<T>& v, cl_mem_flags flags);

  template <typename T>
  void copy_buf_to_vector(const cl::Buffer& buf, size_t count, std::vector<T>* v);

 private:
  cl::Context context;
  cl::Device device;
  cl::CommandQueue queue;
};

template<typename T>
static inline size_t vector_size_bytes(const std::vector<T>& v) {
  return v.size() * sizeof(v[0]);
}

template<typename T>
cl::Buffer ClEnvironment::create_buf_from_vector(const std::vector<T>& v, cl_mem_flags flags) {
  auto buf_size_bytes = vector_size_bytes(v);
  cl::Buffer buf(context, flags, buf_size_bytes);
  queue.enqueueWriteBuffer(buf, CL_TRUE, 0, buf_size_bytes, v.data());
  return buf;
}

template<typename T>
void ClEnvironment::copy_buf_to_vector(const cl::Buffer& buf, size_t count, std::vector<T>* v) {
  v->resize(count);
  queue.enqueueReadBuffer(buf, CL_TRUE, 0, vector_size_bytes(*v), v->data());
}

#endif  // OPENCLLAB01_CL_ENVIRONMENT_HPP
