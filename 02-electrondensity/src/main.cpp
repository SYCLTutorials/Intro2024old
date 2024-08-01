#include "Field.hpp"
#include "Wavefunction.hpp"
#include "version.hpp"
#include "Timer.hpp"
#include <cstdlib>

int main(int argc, char *argv[]) {
  std::cout << "Version: " << PROJECT_VER << std::endl;
  std::cout << "Compilation Date: " << __DATE__ << "  " << __TIME__
            << std::endl;
  std::cout << "Git SHA1: " << GIT_SHA1 << std::endl;

  Wavefunction wf;
  if( argc != 2){
    std::cout << " We need more arguments try with:" << std::endl;
    std::cout << " ./" << argv[0] << " foo.wfx" << std::endl;
    exit(EXIT_FAILURE);
  }
  wf.loadWF(argv[1]);

  Field field(wf);

  Timer tcpu, tgpu, tgpu2;

  tcpu.start();
  field.evalDensity2();
  tcpu.stop();

  tgpu.start();
  field.evalDensity_sycl();
  tgpu.stop();

  tgpu2.start();
  field.evalDensity_sycl2();
  tgpu2.stop();

  std::cout << " Time for CPU : " << tcpu.getDuration() << " \u03BC"
            << "s" << std::endl;

  std::cout << " Time for GPU  : " << tgpu.getDuration() << " \u03BC"
            << "s (Kernel 1)" << std::endl;

  std::cout << " Time for GPU  : " << tgpu2.getDuration() << " \u03BC"
            << "s (Kernel 2)" << std::endl;

  std::cout << " Ratio CPU/GPU (kernel1) : " << tcpu.getDuration() / tgpu.getDuration() << std::endl;
  std::cout << " Ratio CPU/GPU (kernel2) : " << tcpu.getDuration() / tgpu2.getDuration() << std::endl;

  //wf.printWF();
  exit(EXIT_SUCCESS);
}
