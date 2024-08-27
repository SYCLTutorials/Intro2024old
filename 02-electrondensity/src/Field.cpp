#include "Field.hpp"
#include "Atom.hpp"
//#include "Lebedev.hpp"
#include "WaveFunction.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

#include <sycl/sycl.hpp>

Field::Field(Wavefunction &wf) : wf(wf) {}

double Field::DensitySYCL2(int norb, int npri, const int *icnt, const int *vang,
                           const double *r, const double *coor,
                           const double *depris, const double *nocc,
                           const double *coef) {
  double den;
  double x = r[0];
  double y = r[1];
  double z = r[2];

  den = 0.0;
  for (int i = 0; i < norb; i++) {
    double mo = 0.0;
    const int i_prim = i * npri;
    for (int j = 0; j < npri; j++) {
      const int vj = 3 * j;
      const int centroj = 3 * icnt[j];
      const double difx = x - coor[centroj];
      const double dify = y - coor[centroj + 1];
      const double difz = z - coor[centroj + 2];
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

//vamadouble Field::DensitySYCL(int norb, int npri, int *icnt, int *vang, double *r,
//vama                          double *coor, double *depris, double *nocc,
//vama                          double *coef, double *moi) {
//vama  double den;
//vama  double x = r[0];
//vama  double y = r[1];
//vama  double z = r[2];
//vama
//vama  for (int j = 0; j < npri; j++) {
//vama    const int vj = 3 * j;
//vama    const int centroj = 3 * icnt[j];
//vama    const double difx = x - coor[centroj];
//vama    const double dify = y - coor[centroj + 1];
//vama    const double difz = z - coor[centroj + 2];
//vama    const double rr = difx * difx + dify * dify + difz * difz;
//vama
//vama    const double expo = exp(-depris[j] * rr);
//vama    const double lx = vang[vj];
//vama    const double ly = vang[vj + 1];
//vama    const double lz = vang[vj + 2];
//vama    const double facx = pow(difx, lx);
//vama    const double facy = pow(dify, ly);
//vama    const double facz = pow(difz, lz);
//vama
//vama    moi[j] = facx * facy * facz * expo;
//vama  }
//vama
//vama  den = 0.0;
//vama  for (int i = 0; i < norb; i++) {
//vama    double mo = 0.0;
//vama    const int i_prim = i * npri;
//vama    for (int j = 0; j < npri; j++) {
//vama      mo += moi[j] * coef[i_prim + j];
//vama    }
//vama    den += (nocc[i] * mo * mo);
//vama  }
//vama  return den;
//vama}

//#include functioncpu.cpp


//double Field::gaussiana(Rvector r, int mu) {
//  int lx = wf.vang[3 * mu];
//  int ly = wf.vang[3 * mu + 1];
//  int lz = wf.vang[3 * mu + 2];
//  Rvector R(wf.atoms[wf.icntrs[mu]].getCoors());
//
//  double x_part = r.get_x() - R.get_x();
//  double y_part = r.get_y() - R.get_y();
//  double z_part = r.get_z() - R.get_z();
//
//  double diff2 = pow(x_part, 2) + pow(y_part, 2) + pow(z_part, 2);
//
//  double gauss = pow(x_part, lx) * pow(y_part, ly) * pow(z_part, lz);
//
//  gauss *= exp(-wf.depris[mu] * diff2);
//
//  return gauss;
//}

//double Field::orbital(Rvector r, int i) {
//  double orb;
//
//  orb = 0.;
//  for (int mu = 0; mu < wf.npri; mu++)
//    orb += (wf.dcoefs[i * wf.npri + mu] * gaussiana(r, mu));
//
//  return orb;
//}
//double Field::Density(Rvector r) {
//  double den;
//  den = 0;
//  for (int i = 0; i < wf.norb; i++)
//    den += wf.dnoccs[i] * orbital(r, i) * orbital(r, i);
//
//  return den;
//}



void Field::dumpCube(double xmin, double ymin, double zmin, double delta,
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

//vamavoid Field::dumpXYZ(std::string filename) {
//vama  std::ofstream fout(filename);
//vama  if (fout.is_open()) {
//vama
//vama    fout << std::setw(4) << std::fixed << wf.natm << std::endl;
//vama    fout << " File created by handleWF code" << std::endl;
//vama    for (auto atom : wf.atoms) {
//vama      fout << std::setw(4) << std::fixed << atom.getSymbol();
//vama      fout << std::setw(13) << std::setprecision(6) << std::fixed
//vama           << atom.get_x() * 0.529177 << ' ';
//vama      fout << std::setw(13) << std::setprecision(6) << std::fixed
//vama           << atom.get_y() * 0.529177 << ' ';
//vama      fout << std::setw(13) << std::setprecision(6) << std::fixed
//vama           << atom.get_z() * 0.529177;
//vama      fout << std::endl;
//vama    }
//vama    fout.close();
//vama  }
//vama}

//void Field::evalDensity2D() {
//
//  vector<double> field;
//  double xmin = -10.0, xmax = 10.0;
//  double ymin = -10.0, ymax = 10.0;
//  double delta = 0.25;
//
//  int npoints_x = int((xmax - xmin) / delta);
//  int npoints_y = int((ymax - ymin) / delta);
//
//  std::cout << " Points ( " << npoints_x << "," << npoints_y << ")  ";
//  std::cout << " total points : " << npoints_x * npoints_y << std::endl;
//
//  std::ofstream fout("density2d.dat");
//  if (fout.is_open()) {
//
//    for (int i = 0; i < npoints_x; i++) {
//      double x = xmin + i * delta;
//      for (int j = 0; j < npoints_y; j++) {
//        double y = ymin + j * delta;
//
//        Rvector r(x, y, 0.0);
//        double den = Density(r);
//
//        fout << std::setw(13) << std::setprecision(6) << std::fixed
//             << x * 0.529177 << ' ';
//        fout << std::setw(13) << std::setprecision(6) << std::fixed
//             << y * 0.529177 << ' ';
//        fout << std::setw(13) << std::setprecision(6) << std::fixed << den;
//        fout << std::endl;
//      }
//      fout << std::endl;
//    }
//  }
//}

#include "function1d.cpp"
#include "function3d.cpp"
#include "evaldensobj.cpp"

//void Field::spherical(std::string filename) {
//  std::ofstream fout(filename);
//
//  if (!fout.is_open())
//    std::cerr << "Error: The file can be opened." << std::endl;
//
//  int maxnpoints = 10000;
//  int iter = 0;
//  int np = 434;
//  Lebedev Leb(np);
//  std::vector<Rvector> rvecs = Leb.getRvecs();
//  std::vector<double> weights = Leb.getWeights();
//  double den = 100;
//  double denx, deny, denz;
//  double rmin = 0.0;
//  double delta = 0.01;
//  double suma = 0.;
//  double r;
//  std::vector<double> vden;
//  std::vector<double> vr;
//  std::vector<double> xden;
//  std::vector<double> yden;
//  std::vector<double> zden;
//  while (iter < maxnpoints && den > 1.E-7) {
//    r = rmin + iter * delta;
//    suma = 0.;
//    for (int i = 0; i < np; i++) {
//      suma += weights[i] * Density(Lebedev::transform(rvecs[i], r));
//    }
//    denx = Density(Rvector(r, 0., 0.));
//    deny = Density(Rvector(0., r, 0.));
//    denz = Density(Rvector(0., 0., r));
//    den = suma * 4. * M_PI;
//    fout << std::setw(14) << std::setprecision(7) << std::fixed << r << ' ';
//    fout << std::setw(14) << std::setprecision(7) << std::scientific << den
//         << ' ';
//    fout << std::setw(14) << std::setprecision(7) << std::scientific << denx
//         << ' ';
//    fout << std::setw(14) << std::setprecision(7) << std::scientific << deny
//         << ' ';
//    fout << std::setw(14) << std::setprecision(7) << std::scientific << denz
//         << ' ';
//    fout << std::endl;
//
//    vden.push_back(den);
//    vr.push_back(r);
//    xden.push_back(denx);
//    yden.push_back(deny);
//    zden.push_back(denz);
//    iter++;
//  }
//
//  fout.close();
//
//  int last = vr.size() - 1;
//  double integral = pow(vr[0], 2) * vden[0] + pow(vr[last], 2) * vden[last];
//  double integralx = pow(vr[0], 2) * xden[0] + pow(vr[last], 2) * xden[last];
//  double integraly = pow(vr[0], 2) * yden[0] + pow(vr[last], 2) * yden[last];
//  double integralz = pow(vr[0], 2) * zden[0] + pow(vr[last], 2) * zden[last];
//
//  for (int i = 1; i < vr.size() - 1; i++) {
//    double r2 = 2. * vr[i] * vr[i];
//    integral += r2 * vden[i];
//    integralx += r2 * xden[i];
//    integraly += r2 * yden[i];
//    integralz += r2 * zden[i];
//  }
//  integral *= 0.5 * delta;
//  integralx *= 0.5 * delta;
//  integraly *= 0.5 * delta;
//  integralz *= 0.5 * delta;
//
//  std::cout << " Integral Atomica " << std::endl;
//  std::cout << "Angular: " << std::setw(14) << std::setprecision(7)
//            << std::fixed << integral << std::endl;
//  std::cout << "Den  x : " << std::setw(14) << std::setprecision(7)
//            << std::fixed << integralx << std::endl;
//  std::cout << "Den  y : " << std::setw(14) << std::setprecision(7)
//            << std::fixed << integraly << std::endl;
//  std::cout << "Den  z : " << std::setw(14) << std::setprecision(7)
//            << std::fixed << integralz << std::endl;
//}
