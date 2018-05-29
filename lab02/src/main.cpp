#include "defer.hpp"
#include "cl_environment.hpp"

#include <fstream>
#include <iomanip>


struct Scanner {
  explicit Scanner(const std::vector<float>& input) noexcept(false) :
      input(input),
      env(),
      program(env.compile_program("src/scan.cl", std::cerr)),
      calculate(cl::Kernel(program, "scan_hillis_steele_calculate", nullptr)),
      propagate(cl::Kernel(program, "scan_hillis_steele_propagate", nullptr)),
      dev_data(env.create_buf_from_vector(input, CL_MEM_READ_WRITE)) {}

  void do_scan(std::vector<float>* output) {
    do_scan_rec(1);
    env.copy_buf_to_vector(dev_data, input.size(), output);
  }

 private:
  constexpr static size_t BLOCK_SIZE = 256;

  const std::vector<float>& input;
  ClEnvironment env;
  cl::Program program;
  cl::Kernel calculate;
  cl::Kernel propagate;
  cl::Buffer dev_data;

  void do_scan_rec(size_t array_period) {
    if (array_period > input.size()) {
      return;
    }

    calculate.setArg(0, dev_data);
    calculate.setArg(1, input.size());
    calculate.setArg(2, array_period);
    // allocating and passing to the kernel local memory buffers
    calculate.setArg(3, cl::Local(sizeof(input[0]) * BLOCK_SIZE));
    calculate.setArg(4, cl::Local(sizeof(input[0]) * BLOCK_SIZE));
    env.run_kernel_1d(calculate, input.size(), BLOCK_SIZE);

    if (array_period * BLOCK_SIZE <= input.size()) {
      do_scan_rec(array_period * BLOCK_SIZE);

      propagate.setArg(0, dev_data);
      propagate.setArg(1, input.size());
      propagate.setArg(2, array_period);
      env.run_kernel_1d(propagate, input.size(), BLOCK_SIZE);
    }
  }
};

void inclusive_scan(const std::vector<float>& a, std::vector<float>* out) {
  Scanner scanner(a);
  scanner.do_scan(out);
}

int main() {
  using namespace std;

  ifstream ins("input.txt");
  ofstream outs("output.txt");
  outs << fixed << setprecision(3);

  size_t n;
  ins >> n;

  std::vector<float> a(n), res(n);
  for (auto& x : a) {
    ins >> x;
  }

  try {
    inclusive_scan(a, &res);
  } catch (cl::Error const& e) {
    std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    return 1;
  }

  for (auto x : res) {
    outs << x << ' ';
  }
  outs << endl;
}
