#ifndef _FIELD_HPP_
#define _FIELD_HPP_

#include "WaveFunction.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

class Field {
public:
  Field(Wavefunction &wf);

  double orbital(Rvector r, int i);
  double gaussiana(Rvector r, int mu);
  double Density(Rvector r);
  void evalDensity();
  void evalDensity2();
  void evalDensity2D();

  void evalDensity_sycl();
  void evalDensity_sycl2();
  static SYCL_EXTERNAL double DensitySYCL(int, int, int *, int *, double *,
                                          double *, double *, double *,
                                          double *, double *);
  static SYCL_EXTERNAL double DensitySYCL2(int, int, const int *, const int *,
                                           const double *, const double *,
                                           const double *, const double *,
                                           const double *);

  void spherical(std::string fname);

  void dumpXYZ(std::string filename);

//  void dumpCube(double, double, double, double, int, int, int, vector<double>,
//                std::string filename);
  void dumpCube(double xmin, double ymin, double zmin, double delta,
                       int nx, int ny, int nz, vector<double> field,
                       std::string filename) {
    std::ofstream fout(filename);
    if (fout.is_open()) {
  
      fout << "Density" << std::endl;
      fout << "By handleWF project" << std::endl;
      fout << std::setw(5) << std::fixed << wf.natm;
      fout << std::setw(13) << std::setprecision(6) << std::fixed << xmin << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << ymin << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << zmin;
      fout << std::endl;
  
      fout << std::setw(5) << std::fixed << nx;
      fout << std::setw(13) << std::setprecision(6) << std::fixed << delta << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0;
      fout << std::endl;
  
      fout << std::setw(5) << std::fixed << ny;
      fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << delta << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0;
      fout << std::endl;
  
      fout << std::setw(5) << std::fixed << nz;
      fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << delta;
      fout << std::endl;
    }
  
    for (auto atom : wf.atoms) {
      fout << std::setw(5) << std::fixed << atom.get_atnum();
      fout << std::setw(13) << std::setprecision(6) << std::fixed
           << atom.get_charge() << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << atom.get_x()
           << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << atom.get_y()
           << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed << atom.get_z();
      fout << std::endl;
    }
  
    int cnt = 0;
    for (auto value : field) {
      cnt++;
      fout << std::setw(15) << std::setprecision(6) << std::fixed
           << std::scientific << value;
      if (cnt == 6) {
        fout << std::endl;
        cnt = 0;
      }
    }
    if (cnt != 0)
      fout << std::endl;
  
    fout.close();
  }


  void configGrid(int  *npx, int *npy, int *npz, double *delta,
                  double *xmin, double *ymin, double *zmin,
                  size_t *nsize){
  
     (*xmin) = -10.0;
     (*ymin) = -10.0;
     (*zmin) = -10.0;
  
     (*delta) = 0.25;
  
     (*npx) = fabs(2.*(*xmin)) / (*delta);
     (*npy) = fabs(2.*(*ymin)) / (*delta);
     (*npz) = fabs(2.*(*zmin)) / (*delta);
  
     (*nsize) = (*npx) * (*npy) * (*npz);
  
   }


private:
  Wavefunction &wf;
};


#endif
