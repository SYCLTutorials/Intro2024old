#include "Promolecule.hpp"
#include "Atom.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>

// My coef
// std::vector<double> c1 = {
//    0.0000000E+00, 4.6568633E-01, 1.9613583E+01, 1.0682467E+02, 4.9307260E-01,
//    2.5663419E+02, 4.0348178E+02, 6.3406713E+02, 9.5831026E+02, 1.6273443E+01,
//    1.9788394E+03, 3.5289679E+03, 1.0046303E+02, 6.2991738E+03, 8.4532518E+03,
//    1.1107832E+04, 1.4348550E+04, 6.0301596E+02, 8.0816108E+02};
// std::vector<double> c2 = {
//    0.0000000E+00, 4.6568622E-01, 1.9568763E+01, 1.0681231E+02, 5.2439107E+00,
//    1.1256576E+00, 2.2620806E+00, 6.3404543E+02, 8.9583281E+00, 1.3870221E+03,
//    1.9614363E+03, 3.5302178E+03, 4.5960222E+03, 1.5274864E+02, 8.4342635E+03,
//    3.1824165E+02, 4.4165626E+02, 1.8325620E+04, 2.3005524E+04};
// std::vector<double> c3 = {
//    0.0000000E+00, 3.0019989E+00, 1.3983775E+00, 3.3342545E+01, 3.1734662E+02,
//    2.5867085E+02, 4.0837497E+02, 4.6765606E+00, 9.5832030E+02, 1.4029674E+03,
//    2.8279579E+01, 7.4670217E+01, 4.6015044E+03, 6.2957609E+03, 2.2368762E+02,
//    1.1103799E+04, 1.4364194E+04, 1.8305210E+04, 2.3084071E+04};
// std::vector<double> a1 = {
//    0.0000000E+00, 1.9888088E+00, 3.8987496E+00, 6.2028021E+00, 7.2657186E-01,
//    9.7173881E+00, 1.1698438E+01, 1.3677050E+01, 1.5655698E+01, 1.5948420E+00,
//    1.9625025E+01, 2.2086110E+01, 2.6605426E+00, 2.6057308E+01, 2.8120183E+01,
//    3.0186892E+01, 3.2254026E+01, 4.2979610E+00, 4.6484753E+00};
// std::vector<double> a2 = {
//    0.0000000E+00, 1.9888088E+00, 3.8982540E+00, 6.2023193E+00, 4.1739221E+00,
//    8.3058204E-01, 9.6295177E-01, 1.3677045E+01, 1.3642293E+00, 1.7635609E+01,
//    1.9625014E+01, 2.2085863E+01, 2.3998338E+01, 2.9772425E+00, 2.8120238E+01,
//    3.6228839E+00, 3.9540388E+00, 3.4332866E+01, 3.6415201E+01};
// std::vector<double> a3 = {
//    0.0000000E+00, 1.9888228E+00, 1.6012664E+00, 3.8061318E+00, 7.8497277E+00,
//    9.7173599E+00, 1.1698438E+01, 1.1538421E+00, 1.5655807E+01, 1.7635595E+01,
//    1.8462843E+00, 2.4886290E+00, 2.3998352E+01, 2.6057363E+01, 3.2971353E+00,
//    3.0186664E+01, 3.2254046E+01, 3.4332897E+01, 3.6415230E+01};

inline bool is_number(const string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

Promolecule::Promolecule(std::string fname) {
  h = 0.25;
  loadXYZ(fname);
}

void Promolecule::loadXYZ(std::string fname) {
  int natm;
  char tmp[10];
  string s;
  double xt, yt, zt;

  std::vector<double> coorx;
  std::vector<double> coory;
  std::vector<double> coorz;

  string line;
  ifstream finp;

  finp.open(fname.c_str(), std::ios::in);
  if (!finp.good()) {
    cout << " The file [" << fname << "] can't be opened!" << endl;
  }
  finp.seekg(finp.beg);

  getline(finp, line); // read the line with the number of nuclei
  sscanf(line.c_str(), "%d", &natm);
  getline(finp, line); // read the comment line;
  for (int i = 0; i < natm; i++) {
    getline(finp, line); // read the line with information of centres
    sscanf(line.c_str(), "%s %lf %lf %lf", tmp, &xt, &yt, &zt);

    xt *= 1.8897259886;
    yt *= 1.8897259886;
    zt *= 1.8897259886;

    coorx.push_back(xt);
    coory.push_back(yt);
    coorz.push_back(zt);

    std::sort(coorx.begin(), coorx.end());
    std::sort(coory.begin(), coory.end());
    std::sort(coorz.begin(), coorz.end());
    double xmin = coorx.front() - 3.;
    double xmax = coorx.back() + 3.;
    double ymin = coory.front() - 3.;
    double ymax = coory.back() + 3.;
    double zmin = coorz.front() - 3.;
    double zmax = coorz.back() + 3.;

    nx = static_cast<int>(std::ceil((xmax - xmin) / h));
    ny = static_cast<int>(std::ceil((ymax - ymin) / h));
    nz = static_cast<int>(std::ceil((zmax - zmin) / h));
    x0 = xmin;
    y0 = ymin;
    z0 = zmin;

    if (!is_number(tmp)) {
      s = string(tmp);
      s.erase(std::remove_if(s.begin(), s.end(),
                             [](char ch) { return std::isdigit(ch); }),
              s.end());
      atoms.push_back(Atom(s, xt, yt, zt));
    } else {
      atoms.push_back(Atom(atoi(tmp), xt, yt, zt));
    }
  }
  natom = atoms.size();

  finp.close();
}

void Promolecule::test() {
  std::cout << " Test of load XYZ" << std::endl;
  for (auto atom : atoms) {
    std::cout << setw(10) << atom.getSymbol();
    std::cout << setw(10) << fixed << setprecision(6) << atom.get_x();
    std::cout << setw(10) << fixed << setprecision(6) << atom.get_y();
    std::cout << setw(10) << fixed << setprecision(6) << atom.get_z()
              << std::endl;
  }
}

double Promolecule::density(double x, double y, double z,
                            std::vector<Atom> atoms) {

  int iatm;
  double xX, yY, zZ;
  double r;

  double rho = 0.;
  for (auto atom : atoms) {
    iatm = atom.get_atnum();
    xX = x - atom.get_x();
    yY = y - atom.get_y();
    zZ = z - atom.get_z();

    r = sqrt(xX * xX + yY * yY + zZ * zZ);

    rho += c1[iatm] * exp(-a1[iatm] * r) + c2[iatm] * exp(-a2[iatm] * r) +
           c3[iatm] * exp(-a3[iatm] * r);
  }

  return rho;
}

double Promolecule::reduced(double x, double y, double z,
                            std::vector<Atom> atoms) {

  int iatm;
  double xX, yY, zZ;
  double r, r1, tmp;

  double rho = 0.;
  double gradx = 0.;
  double grady = 0.;
  double gradz = 0.;
  for (auto atom : atoms) {
    iatm = atom.get_atnum();
    xX = x - atom.get_x();
    yY = y - atom.get_y();
    zZ = z - atom.get_z();

    r = sqrt(xX * xX + yY * yY + zZ * zZ);
    r1 = 1. / r;

    rho += c1[iatm] * exp(-a1[iatm] * r) + c2[iatm] * exp(-a2[iatm] * r) +
           c3[iatm] * exp(-a3[iatm] * r);

    tmp = -a1[iatm] * c1[iatm] * exp(-a1[iatm] * r) +
          -a2[iatm] * c2[iatm] * exp(-a2[iatm] * r) +
          -a3[iatm] * c3[iatm] * exp(-a3[iatm] * r);

    gradx += tmp * xX * r1;
    grady += tmp * yY * r1;
    gradz += tmp * zZ * r1;
  }

  tmp = sqrt(gradx * gradx + grady * grady + gradz * gradz);

  return tmp * 0.1616204596739955 * pow(rho, (-4. / 3.));
}

std::vector<double> Promolecule::hessian(double x, double y, double z,
                                         std::vector<Atom> atoms) {

  int iatm;
  double xX, yY, zZ;
  double xX2, yY2, zZ2;
  double r, r1, r2, r13;
  double hesxx, hesyy, heszz;
  double hesxy, hesxz, hesyz;
  hesxx = 0.;
  hesxy = 0.;
  hesxz = 0.;
  hesyy = 0.;
  hesyz = 0.;
  heszz = 0.;
  for (auto atom : atoms) {
    iatm = atom.get_atnum();
    xX = x - atom.get_x();
    yY = y - atom.get_y();
    zZ = z - atom.get_z();

    xX2 = xX * xX;
    yY2 = yY * yY;
    zZ2 = zZ * zZ;

    r2 = xX2 + yY2 + zZ2;
    r = sqrt(r2);
    r1 = 1. / r;
    r13 = pow(r1, 3.);
    auto e1 = exp(-a1[iatm] * r);
    auto e2 = exp(-a1[iatm] * r);
    auto e3 = exp(-a1[iatm] * r);

    hesxx += a1[iatm] * c1[iatm] * e1 * (xX2 * (a1[iatm] * r + 1) - r2) * r13 +
             a2[iatm] * c2[iatm] * e2 * (xX2 * (a2[iatm] * r + 1) - r2) * r13 +
             a3[iatm] * c3[iatm] * e3 * (xX2 * (a3[iatm] * r + 1) - r2) * r13;

    hesyy += a1[iatm] * c1[iatm] * e1 * (yY2 * (a1[iatm] * r + 1) - r2) * r13 +
             a2[iatm] * c2[iatm] * e2 * (yY2 * (a2[iatm] * r + 1) - r2) * r13 +
             a3[iatm] * c3[iatm] * e3 * (yY2 * (a3[iatm] * r + 1) - r2) * r13;

    heszz += a1[iatm] * c1[iatm] * e1 * (zZ2 * (a1[iatm] * r + 1) - r2) * r13 +
             a2[iatm] * c2[iatm] * e2 * (zZ2 * (a2[iatm] * r + 1) - r2) * r13 +
             a3[iatm] * c3[iatm] * e3 * (zZ2 * (a3[iatm] * r + 1) - r2) * r13;

    hesxy += a1[iatm] * c1[iatm] * e1 * (xX * yY * (a1[iatm] * r + 1)) * r13 +
             a2[iatm] * c2[iatm] * e2 * (xX * yY * (a2[iatm] * r + 1)) * r13 +
             a3[iatm] * c3[iatm] * e3 * (xX * yY * (a3[iatm] * r + 1)) * r13;

    hesxz += a1[iatm] * c1[iatm] * e1 * (xX * zZ * (a1[iatm] * r + 1)) * r13 +
             a2[iatm] * c2[iatm] * e2 * (xX * zZ * (a2[iatm] * r + 1)) * r13 +
             a3[iatm] * c3[iatm] * e3 * (xX * zZ * (a3[iatm] * r + 1)) * r13;

    hesyz += a1[iatm] * c1[iatm] * e1 * (yY * zZ * (a1[iatm] * r + 1)) * r13 +
             a2[iatm] * c2[iatm] * e2 * (yY * zZ * (a2[iatm] * r + 1)) * r13 +
             a3[iatm] * c3[iatm] * e3 * (yY * zZ * (a3[iatm] * r + 1)) * r13;
  }

  std::vector<double> hess(6);
  hess[0] = hesxx;
  hess[1] = hesxy;
  hess[2] = hesxz;
  hess[3] = hesyy;
  hess[4] = hesyz;
  hess[5] = heszz;

  return hess;
}

double Promolecule::getLambda2(std::vector<double> hess) {
  double a = hess[0];
  double b = hess[1];
  double c = hess[2];
  double d = hess[3];
  double e = hess[4];
  double g = hess[5];
  double p, q, r, detB, phi;
  std::vector<double> eval(3);

  p = b * b + c * c + e * e;
  if (p == 0) {
    eval[0] = a;
    eval[1] = d;
    eval[2] = g;
  } else {
    q = (a + d + g) / 3.;
    p = (a - q) * (a - q) + (d - q) * (d - q) + (g - q) * (g - q) + 2. * p;
    p = sqrt(p / 6.);
    detB = (a - q) * ((d - q) * (g - q) - e * e);
    detB = detB - b * (b * (g - q) - e * c);
    detB = detB + c * (b * e - c * (d - q));
    detB = detB / (p * p * p);
    r = detB / 2.;
    if (r <= -1.)
      phi = M_PI / 3.;
    else {
      if (r >= 1)
        phi = 0.;
      else
        phi = acos(r) / 3.;
    }
    eval[0] = q + 2. * p * cos(phi);
    eval[2] = q + 2. * p * cos(phi + M_PI * (2. / 3.));
    eval[1] = 3. * q - eval[0] - eval[2];
  }
  std::sort(eval.begin(), eval.end());

  return eval[1];
}

void Promolecule::evalNCI(std::string name) {
  double x, y, z;
  double rho, red;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        x = x0 + i * h;
        y = y0 + j * h;
        z = z0 + k * h;

        rho = density(x, y, z, atoms);

        if (rho < 0.05 && rho > 1.E-6) {
          red = reduced(x, y, z, atoms);
          auto hess = hessian(x, y, z, atoms);
          rho = getLambda2(hess) * rho * 100;
          rdg.push_back(red);
          den.push_back(rho);
        } else {
          red = 100.;
          rho = 100.;
          rdg.push_back(red);
          den.push_back(rho);
        }
      }
    }
  }
  dumpCube(den, name + "Den.cube");
  dumpCube(rdg, name + "Red.cube");
  dumpData(name + ".dat");
}

void Promolecule::dumpCube(std::vector<double> field, std::string filename) {
  std::ofstream fout(filename);
  if (fout.is_open()) {

    fout << "Density" << std::endl;
    fout << "By handleWF project" << std::endl;
    fout << std::setw(5) << std::fixed << natom;
    fout << std::setw(13) << std::setprecision(6) << std::fixed << x0 << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << y0 << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << z0;
    fout << std::endl;

    fout << std::setw(5) << std::fixed << nx;
    fout << std::setw(13) << std::setprecision(6) << std::fixed << h << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0;
    fout << std::endl;

    fout << std::setw(5) << std::fixed << ny;
    fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << h << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0;
    fout << std::endl;

    fout << std::setw(5) << std::fixed << nz;
    fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << 0.0 << ' ';
    fout << std::setw(13) << std::setprecision(6) << std::fixed << h;
    fout << std::endl;
  }

  for (auto atom : atoms) {
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
  for (auto valor : field) {
    cnt++;
    fout << std::setw(15) << std::setprecision(6) << std::fixed
         << std::scientific << valor;
    if (cnt == 6) {
      fout << std::endl;
      cnt = 0;
    }
  }
  if (cnt != 0)
    fout << std::endl;

  fout.close();
}

void Promolecule::dumpData(std::string filename) {
  std::ofstream fout(filename);
  if (fout.is_open()) {

    fout << "# Density and Reduced" << std::endl;
    fout << "# By handleWF project" << std::endl;

    for (int i = 0; i < den.size(); i++) {
      if (fabs(den[i]) < 0.05 * 100.) {
        fout << std::setw(15) << std::setprecision(6) << std::fixed
             << std::scientific << den[i] << "  ";
        fout << std::setw(15) << std::setprecision(6) << std::fixed
             << std::scientific << rdg[i];
        fout << std::endl;
      }
    }
  }
  fout.close();
}