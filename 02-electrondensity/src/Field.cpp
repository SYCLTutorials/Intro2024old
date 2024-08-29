#include "Field.hpp"
#include "Atom.hpp"
#include "WaveFunction.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

#include <sycl/sycl.hpp>

Field::Field(Wavefunction &wf, double rmin, double delta) : wf(wf), xmin(rmin), ymin(rmin), zmin(rmin), delta(delta){

    npoints_x = static_cast<int>(fabs(2.*xmin / delta));
    npoints_y = static_cast<int>(fabs(2.*ymin / delta));
    npoints_z = static_cast<int>(fabs(2.*zmin / delta));

    nsize = npoints_x * npoints_y * npoints_z;
}

double Field::DensitySYCL2(int norb, int npri, const int *icnt, const int *vang,
                           const double *r, const double *coor,
                           const double *depris, const double *nocc,
                           const double *coef) {
  double den = 0.0;
  const double x = r[0];
  const double y = r[1];
  const double z = r[2];

  for (int i = 0; i < norb; i++) {
    double mo = 0.0;
    const int i_prim = i * npri;
    for (int j = 0; j < npri; j++) {
      const int vj = 3 * j;
      const int centerj = 3 * icnt[j];
      const double difx = x - coor[centerj];
      const double dify = y - coor[centerj + 1];
      const double difz = z - coor[centerj + 2];
      const double rr = difx * difx + dify * dify + difz * difz;

      const double expo = exp(-depris[j] * rr);
      const double lx = vang[vj];
      const double ly = vang[vj + 1];
      const double lz = vang[vj + 2];
      const double facx = pow(difx, lx);
      const double facy = pow(dify, ly);
      const double facz = pow(difz, lz);

      mo += facx * facy * facz * expo * coef[i_prim + j];
    }
    den += nocc[i] * mo * mo;
  }

  return den;
}


#include "functioncpu.xx"

//#include "function1d.cxx"
//#include "function3d.cxx"
//#include "evaldensobj.cxx"





