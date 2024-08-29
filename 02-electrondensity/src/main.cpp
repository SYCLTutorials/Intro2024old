#include "Field.hpp"
#include "WaveFunction.hpp"
#include "version.hpp"
#include "Timer.hpp"
#include <cstdlib>

int main(int argc, char *argv[]) {
  std::cout << "Version: " << PROJECT_VER << std::endl;
  std::cout << "Compilation Date: " << __DATE__ << "  " << __TIME__
            << std::endl;
  std::cout << "Git SHA1: " << GIT_SHA1 << std::endl;

  Wavefunction wf;
  if( argc != 4){
    std::cout << " We need more arguments try with:" << std::endl;
    std::cout << " ./" << argv[0] << " foo.wfx"  << " delta" << " rmin"<< std::endl;
    exit(EXIT_FAILURE);
  }

  wf.loadWF(argv[1]);
  double delta = std::stod(argv[2]);
  double rmin  = std::stod(argv[3]);

  Field field(wf, delta, rmin);

  Timer tcpu, tgpu, tgpu2;

  tcpu.start();
  field.evalDensity2();
  tcpu.stop();
//vama
//vama  tgpu.start();
//vama  field.evalDensity_sycl();
//vama  tgpu.stop();
//vama
//vama  tgpu2.start();
//vama  field.evalDensity_sycl2();
//vama  tgpu2.stop();
//vama
//vama  std::cout << " Time for CPU : " << tcpu.getDuration() << " \u03BC"
//vama            << "s" << std::endl;
//vama
//vama  std::cout << " Time for GPU  : " << tgpu.getDuration() << " \u03BC"
//vama            << "s (Kernel 1)" << std::endl;
//vama
//vama  std::cout << " Time for GPU  : " << tgpu2.getDuration() << " \u03BC"
//vama            << "s (Kernel 2)" << std::endl;
//vama
//vama  std::cout << " Ratio CPU/GPU (kernel1) : " << tcpu.getDuration() / tgpu.getDuration() << std::endl;
//vama  std::cout << " Ratio CPU/GPU (kernel2) : " << tcpu.getDuration() / tgpu2.getDuration() << std::endl;
//vama
//vama  //wf.printWF();
//vama  exit(EXIT_SUCCESS);
}
