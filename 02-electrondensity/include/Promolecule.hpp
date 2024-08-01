#ifndef _PROMOLECULE_HPP_
#define _PROMOLECULE_HPP_

#include "Atom.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

class Promolecule {
public:
  Promolecule(std::string fname);
  void loadXYZ(std::string fname);
  void test();

  void dumpCube(vector<double>, std::string filename);

  void dumpData(std::string filename);

  void evalNCI(std::string name);

private:
  int natom;
  int nx, ny, nz;
  double x0, y0, z0;
  double h;
  std::vector<double> c1 = {
      0.000000e+00, 2.815000e-01, 2.437000e+00, 1.184000e+01, 3.134000e+01,
      6.782000e+01, 1.202000e+02, 1.909000e+02, 2.895000e+02, 4.063000e+02,
      5.613000e+02, 7.608000e+02, 1.016000e+03, 1.319000e+03, 1.658000e+03,
      2.042000e+03, 2.501000e+03, 3.024000e+03, 3.625000e+03};
  std::vector<double> c2 = {
      0.000000e+00, 0.000000e+00, 0.000000e+00, 6.332000e-02, 3.694000e-01,
      8.527000e-01, 1.172000e+00, 2.247000e+00, 2.879000e+00, 3.049000e+00,
      6.984000e+00, 2.242000e+01, 3.717000e+01, 5.795000e+01, 8.716000e+01,
      1.157000e+02, 1.580000e+02, 2.055000e+02, 2.600000e+02};
  std::vector<double> c3 = {
      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
      0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
      0.000000e+00, 6.358000e-02, 3.331000e-01, 8.878000e-01, 7.888000e-01,
      1.465000e+00, 2.170000e+00, 3.369000e+00, 5.211000e+00};
  std::vector<double> a1 = {
      1.000000e+00, 1.891074e+00, 2.959455e+00, 5.230126e+00, 7.194245e+00,
      9.442871e+00, 1.131222e+01, 1.303781e+01, 1.494768e+01, 1.644737e+01,
      1.821494e+01, 2.016129e+01, 2.227171e+01, 2.433090e+01, 2.617801e+01,
      2.793296e+01, 2.985075e+01, 3.174603e+01, 3.378378e+01};
  std::vector<double> a2 = {
      1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000801e+00, 1.439885e+00,
      1.886792e+00, 1.824818e+00, 2.206531e+00, 2.516356e+00, 2.503756e+00,
      2.901073e+00, 3.982477e+00, 4.651163e+00, 5.336179e+00, 6.045949e+00,
      6.626905e+00, 7.304602e+00, 7.942812e+00, 8.561644e+00};
  std::vector<double> a3 = {
      1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00,
      1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00,
      1.000000e+00, 9.769441e-01, 1.289823e+00, 1.677290e+00, 1.429593e+00,
      1.709110e+00, 1.942125e+00, 2.010454e+00, 2.266546e+00};

  std::vector<Atom> atoms;
  std::vector<double> den;
  std::vector<double> rdg;
  double density(double x, double y, double z, std::vector<Atom> atoms);
  double reduced(double x, double y, double z, std::vector<Atom> atoms);
  std::vector<double> hessian(double x, double y, double z,
                              std::vector<Atom> atoms);
  double getLambda2(std::vector<double>);
};

#endif
