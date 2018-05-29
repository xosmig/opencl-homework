#include "cl_environment.hpp"

#include "defer.hpp"
#include <fstream>
#include <string>
#include <cassert>

ClEnvironment::ClEnvironment() {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;

  // create platform
  cl::Platform::get(&platforms);
  platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
  if (devices.empty()) {
    throw std::domain_error("No devices found.");
  }
  this->device = devices[0];

  // create context
  this->context = cl::Context(devices);

  // initialize queue
  this->queue = cl::CommandQueue(this->context, this->device, CL_QUEUE_PROFILING_ENABLE);
}

cl::Program ClEnvironment::compile_program(const std::string& file_name, std::ostream& log) {
  // load opencl source
  std::ifstream cl_file(file_name);
  std::string cl_string(std::istreambuf_iterator<char>(cl_file),
                        (std::istreambuf_iterator<char>()));
  cl::Program::Sources sources(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

  // create program
  cl::Program program(this->context, sources);

  // compile opencl source
  auto print_log = [&]() {
    log << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device) << std::endl;
  };
  try {
    program.build(std::vector<cl::Device>{this->device});
  } catch (...) {
    print_log();
    throw;
  }
  print_log();

  return std::move(program);
}

void ClEnvironment::run_kernel_1d(const cl::Kernel& kernel, size_t global_size, size_t block_size) {
  auto global_size_rounded =  ((global_size + block_size - 1) / block_size) * block_size;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                             cl::NDRange(global_size_rounded), cl::NDRange(block_size));
}
