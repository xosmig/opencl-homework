#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <memory>

#include <CL/cl.hpp>


struct ClEnvironment {
  cl::Context context;
  cl::Kernel kernel;
  cl::Device device;
  cl::CommandQueue queue;
};

typedef std::shared_ptr<void> Defer;

template <typename F>
Defer defer(F f) {
  return Defer(nullptr, [f{std::move(f)}](...) { f(); });
}

void build_kernel(const std::string& file_name, const std::string& kernel_name,
                  ClEnvironment* out, std::ostream& log) {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;

  // create platform
  cl::Platform::get(&platforms);
  platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
  if (devices.empty()) {
    throw std::domain_error("No devices found.");
  }
  out->device = devices[0];

  // create context
  out->context = cl::Context(devices);

  // load opencl source
  std::ifstream cl_file(file_name);
  std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
  cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

  // create program
  cl::Program program(out->context, source);

  // compile opencl source
  {
    auto defer_print_log = defer([&]() {
      log << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    });
    program.build(devices);
  }

  out->kernel = cl::Kernel(program, kernel_name.c_str());

  out->queue = cl::CommandQueue(out->context, out->device, CL_QUEUE_PROFILING_ENABLE);
}

class Matrix {
 public:
  Matrix(size_t size): size_(size), data_(size_ * size_, 0) {}

  size_t size() const { return size_; }
  size_t data_size() const { return data_.size(); }

  float* data() { return data_.data(); }
  const float* data() const {return data_.data(); }

  float* operator[](size_t row) { return data() + size() * row; }
  const float* operator[](size_t row) const { return data() + size() * row;  }

  size_t byte_size() const {
    return sizeof(*data()) * data_size();
  }

  cl::Buffer cl_read_buffer(const ClEnvironment& env) const {
    cl::Buffer res(env.context, CL_MEM_READ_ONLY, byte_size());
    env.queue.enqueueWriteBuffer(res, CL_TRUE, 0, byte_size(), data());
    return res;
  }

  cl::Buffer cl_write_buffer(const ClEnvironment& env) const {
    return cl::Buffer(env.context, CL_MEM_WRITE_ONLY, byte_size());
  }

  void read_from_cl_buffer(const ClEnvironment& env, const cl::Buffer& buf) {
    env.queue.enqueueReadBuffer(buf, CL_TRUE, 0, byte_size(), data());
  }

  void read(std::istream& ins) {
    for (size_t row = 0; row < size(); ++row) {
      for (size_t col = 0; col < size(); ++col) {
        ins >> (*this)[row][col];
      }
    }
  }

  void print(std::ostream& outs) const {
    for (size_t row = 0; row < size(); ++row) {
      for (size_t col = 0; col < size(); ++col) {
        outs << (*this)[row][col] << ' ';
      }
      outs << std::endl;
    }
  }

 private:
  size_t size_;
  std::vector<float> data_;
};

void conv(const Matrix& m, const Matrix& coef, Matrix* out, size_t block_size = 16) {
  if (block_size < coef.size()) {
    throw std::invalid_argument("block_size must be >= coef.size()");
  }

  ClEnvironment env;
  build_kernel("src/conv.cl", "conv", &env, std::cerr);

  auto dev_m = m.cl_read_buffer(env);
  auto dev_coef = coef.cl_read_buffer(env);
  auto dev_out = out->cl_write_buffer(env);

  env.kernel.setArg(0, dev_m);
  env.kernel.setArg(1, m.size());
  env.kernel.setArg(2, dev_coef);
  env.kernel.setArg(3, coef.size());
  env.kernel.setArg(4, dev_out);

  size_t size_rounded = (std::max(m.size(), coef.size()) + block_size - 1) / block_size * block_size;
  env.queue.enqueueNDRangeKernel(env.kernel,
    cl::NullRange, cl::NDRange(size_rounded, size_rounded), cl::NDRange(block_size, block_size));

  out->read_from_cl_buffer(env, dev_out);
}

int main() {
  using namespace std;

  ifstream ins("input.txt");
  ofstream outs("output.txt");
  outs << fixed << setprecision(3);

  size_t m_size, coef_size;
  ins >> m_size >> coef_size;

  Matrix m(m_size), coef(coef_size), res(m_size);

  m.read(ins);
  coef.read(ins);

  try {
    conv(m, coef, &res);
  } catch (cl::Error const & e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    return 1;
  }

  res.print(outs);
}
