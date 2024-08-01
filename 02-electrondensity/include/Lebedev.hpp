#ifndef _LEBEDEV_HPP_
#define _LEBEDEV_HPP_

#include "Rvector.hpp"
#include <cmath>
#include <iostream>
#include <vector>

class Lebedev {
public:
  Lebedev(int np);

  std::vector<Rvector> &getRvecs();
  std::vector<double> &getWeights();
  static Rvector transform(Rvector rvec, double newr);

private:
  int npoints;
  int getCorrectNPoint(int np);
  std::vector<Rvector> rvecs;
  std::vector<double> weights;

  void genGroup01(double, double, double);
  void genGroup02(double, double, double);
  void genGroup03(double, double, double);
  void genGroup04(double, double, double);
  void genGroup05(double, double, double);
  void genGroup06(double, double, double);

  void Lebedev0006();
  void Lebedev0014();
  void Lebedev0026();
  void Lebedev0038();
  void Lebedev0050();
  void Lebedev0074();
  void Lebedev0086();
  void Lebedev0110();
  void Lebedev0146();
  void Lebedev0170();
  void Lebedev0194();
  void Lebedev0230();
  void Lebedev0266();
  void Lebedev0302();
  void Lebedev0350();
  void Lebedev0434();
};

#endif
