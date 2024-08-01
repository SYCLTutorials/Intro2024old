#include "Field.hpp"
#include "Atom.hpp"
#include "Lebedev.hpp"
#include "Wavefunction.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

#include <sycl/sycl.hpp>

Field::Field(Wavefunction &wf) : wf(wf) {}

double Field::DensitySYCL(int norb, int npri, int *icnt, int *vang, double *r,
                          double *coor, double *depris, double *nocc,
                          double *coef, double *moi) {
  double den;
  double x = r[0];
  double y = r[1];
  double z = r[2];

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

    moi[j] = facx * facy * facz * expo;
  }

  den = 0.0;
  for (int i = 0; i < norb; i++) {
    double mo = 0.0;
    const int i_prim = i * npri;
    for (int j = 0; j < npri; j++) {
      mo += moi[j] * coef[i_prim + j];
    }
    den += (nocc[i] * mo * mo);
  }
  return den;
}

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

double Field::gaussiana(Rvector r, int mu) {
  int lx = wf.vang[3 * mu];
  int ly = wf.vang[3 * mu + 1];
  int lz = wf.vang[3 * mu + 2];
  Rvector R(wf.atoms[wf.icntrs[mu]].getCoors());

  double x_part = r.get_x() - R.get_x();
  double y_part = r.get_y() - R.get_y();
  double z_part = r.get_z() - R.get_z();

  double diff2 = pow(x_part, 2) + pow(y_part, 2) + pow(z_part, 2);

  double gauss = pow(x_part, lx) * pow(y_part, ly) * pow(z_part, lz);

  gauss *= exp(-wf.depris[mu] * diff2);

  return gauss;
}

double Field::orbital(Rvector r, int i) {
  double orb;

  orb = 0.;
  for (int mu = 0; mu < wf.npri; mu++)
    orb += (wf.dcoefs[i * wf.npri + mu] * gaussiana(r, mu));

  return orb;
}
double Field::Density(Rvector r) {
  double den;
  den = 0;
  for (int i = 0; i < wf.norb; i++)
    den += wf.dnoccs[i] * orbital(r, i) * orbital(r, i);

  return den;
}

void Field::evalDensity() {

  vector<double> field;
  double xmin = -10.0, xmax = 10.0;
  double ymin = -10.0, ymax = 10.0;
  double zmin = -5.0, zmax = 5.0;
  double delta = 0.25;

  int npoints_x = int((xmax - xmin) / delta);
  int npoints_y = int((ymax - ymin) / delta);
  int npoints_z = int((zmax - zmin) / delta);

  std::cout << " Points ( " << npoints_x << "," << npoints_y << "," << npoints_z
            << ")" << std::endl;
  std::cout << " TotalPoints : " << npoints_x * npoints_y * npoints_z
            << std::endl;

  for (int i = 0; i < npoints_x; i++) {
    double x = xmin + i * delta;
    for (int j = 0; j < npoints_y; j++) {
      double y = ymin + j * delta;
      for (int k = 0; k < npoints_z; k++) {
        double z = zmin + k * delta;
        Rvector r(x, y, z);
        double den = Density(r);

        field.push_back(den);
      }
    }
  }

  dumpCube(xmin, ymin, zmin, delta, npoints_x, npoints_y, npoints_z, field,
           "densityCPU.cube");
  dumpXYZ("structure.xyz");
}

void Field::evalDensity2() {

  vector<double> field;
  double xmin = -10.0, xmax = 10.0;
  double ymin = -10.0, ymax = 10.0;
  double zmin = -5.0, zmax = 5.0;
  double delta = 0.25;

  int npoints_x = int((xmax - xmin) / delta);
  int npoints_y = int((ymax - ymin) / delta);
  int npoints_z = int((zmax - zmin) / delta);

  double *coor = new double[3 * wf.natm];
  for (int i = 0; i < wf.natm; i++) {
    Rvector R(wf.atoms[i].getCoors());
    coor[3 * i] = R.get_x();
    coor[3 * i + 1] = R.get_y();
    coor[3 * i + 2] = R.get_z();
  }

  std::cout << " Points ( " << npoints_x << "," << npoints_y << "," << npoints_z
            << ")" << std::endl;
  std::cout << " TotalPoints : " << npoints_x * npoints_y * npoints_z
            << std::endl;

  for (int i = 0; i < npoints_x; i++) {
    double x = xmin + i * delta;
    for (int j = 0; j < npoints_y; j++) {
      double y = ymin + j * delta;
      for (int k = 0; k < npoints_z; k++) {
        double z = zmin + k * delta;
        double r[3];
        r[0] = x;
        r[1] = y;
        r[2] = z;

        double den = DensitySYCL2(wf.norb, wf.npri, wf.icntrs.data(),
                                  wf.vang.data(), r, coor, wf.depris.data(),
                                  wf.dnoccs.data(), wf.dcoefs.data());

        field.push_back(den);
      }
    }
  }

  dumpCube(xmin, ymin, zmin, delta, npoints_x, npoints_y, npoints_z, field,
           "densityCPU.cube");
  dumpXYZ("structure.xyz");

  delete[] coor;
}
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

void Field::dumpXYZ(std::string filename) {
  std::ofstream fout(filename);
  if (fout.is_open()) {

    fout << std::setw(4) << std::fixed << wf.natm << std::endl;
    fout << " File created by handleWF code" << std::endl;
    for (auto atom : wf.atoms) {
      fout << std::setw(4) << std::fixed << atom.getSymbol();
      fout << std::setw(13) << std::setprecision(6) << std::fixed
           << atom.get_x() * 0.529177 << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed
           << atom.get_y() * 0.529177 << ' ';
      fout << std::setw(13) << std::setprecision(6) << std::fixed
           << atom.get_z() * 0.529177;
      fout << std::endl;
    }
    fout.close();
  }
}

void Field::evalDensity2D() {

  vector<double> field;
  double xmin = -10.0, xmax = 10.0;
  double ymin = -10.0, ymax = 10.0;
  double delta = 0.25;

  int npoints_x = int((xmax - xmin) / delta);
  int npoints_y = int((ymax - ymin) / delta);

  std::cout << " Points ( " << npoints_x << "," << npoints_y << ")  ";
  std::cout << " total points : " << npoints_x * npoints_y << std::endl;

  std::ofstream fout("density2d.dat");
  if (fout.is_open()) {

    for (int i = 0; i < npoints_x; i++) {
      double x = xmin + i * delta;
      for (int j = 0; j < npoints_y; j++) {
        double y = ymin + j * delta;

        Rvector r(x, y, 0.0);
        double den = Density(r);

        fout << std::setw(13) << std::setprecision(6) << std::fixed
             << x * 0.529177 << ' ';
        fout << std::setw(13) << std::setprecision(6) << std::fixed
             << y * 0.529177 << ' ';
        fout << std::setw(13) << std::setprecision(6) << std::fixed << den;
        fout << std::endl;
      }
      fout << std::endl;
    }
  }
}

void Field::evalDensity_sycl() {

  sycl::queue q(sycl::default_selector_v);
  std::cout << " Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  double xmin = -10.0, xmax = 10.0;
  double ymin = -10.0, ymax = 10.0;
  double zmin = -5.0, zmax = 5.0;
  double delta = 0.25;
  vector<double> field;

  int npoints_x = int((xmax - xmin) / delta);
  int npoints_y = int((ymax - ymin) / delta);
  int npoints_z = int((zmax - zmin) / delta);
  const size_t nsize = npoints_x * npoints_y * npoints_z;
  int natm = wf.natm;
  int npri = wf.npri;
  int norb = wf.norb;
  double *field_local = new double[nsize];

  std::cout << " Points ( " << npoints_x << "," << npoints_y << "," << npoints_z
            << ")" << std::endl;
  std::cout << " TotalPoints : " << npoints_x * npoints_y * npoints_z
            << std::endl;

  double *coor = new double[3 * natm];
  for (int i = 0; i < natm; i++) {
    Rvector R(wf.atoms[i].getCoors());
    coor[3 * i] = R.get_x();
    coor[3 * i + 1] = R.get_y();
    coor[3 * i + 2] = R.get_z();
  }
  // Here we start the sycl kernel
  {
    sycl::buffer<int, 1> icnt_buff(wf.icntrs.data(), sycl::range<1>(npri));
    sycl::buffer<int, 1> vang_buff(wf.vang.data(), sycl::range<1>(3 * npri));
    sycl::buffer<double, 1> coor_buff(coor, sycl::range<1>(3 * natm));
    sycl::buffer<double, 1> eprim_buff(wf.depris.data(), sycl::range<1>(npri));
    sycl::buffer<double, 1> coef_buff(wf.dcoefs.data(),
                                      sycl::range<1>(npri * norb));
    sycl::buffer<double, 1> nocc_buff(wf.dnoccs.data(), sycl::range<1>(norb));
    sycl::buffer<double, 1> field_buff(field_local, sycl::range<1>(nsize));

    q.submit([&](sycl::handler &h) {
      auto field_acc = field_buff.get_access<sycl::access::mode::write>(h);
      auto icnt_acc = icnt_buff.get_access<sycl::access::mode::read>(h);
      auto vang_acc = vang_buff.get_access<sycl::access::mode::read>(h);
      auto coor_acc = coor_buff.get_access<sycl::access::mode::read>(h);
      auto eprim_acc = eprim_buff.get_access<sycl::access::mode::read>(h);
      auto coef_acc = coef_buff.get_access<sycl::access::mode::read>(h);
      auto nocc_acc = nocc_buff.get_access<sycl::access::mode::read>(h);

      h.parallel_for<class Field2>(sycl::range<1>(nsize), [=](sycl::id<1> idx) {
        double cart[3];
        int k = (int)idx % npoints_z;
        int j = ((int)idx / npoints_z) % npoints_y;
        int i = (int)idx / (npoints_z * npoints_y);

        cart[0] = xmin + i * delta;
        cart[1] = ymin + j * delta;
        cart[2] = zmin + k * delta;

        const int *icnt_ptr =
            icnt_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
        const int *vang_ptr =
            vang_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
        const double *coor_ptr =
            coor_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
        const double *eprim_ptr =
            eprim_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
        const double *nocc_ptr =
            nocc_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
        const double *coef_ptr =
            coef_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

        field_acc[idx] = DensitySYCL2(norb, npri, icnt_ptr, vang_ptr, cart,
                                      coor_ptr, eprim_ptr, nocc_ptr, coef_ptr);
      });
    });
    q.wait();
  }
  // End the kernel of SYCL

  for (int i = 0; i < nsize; i++)
    field.push_back(field_local[i]);

  dumpCube(xmin, ymin, zmin, delta, npoints_x, npoints_y, npoints_z, field,
           "densitySYCL1.cube");
  dumpXYZ("structure.xyz");

  delete[] coor;
  delete[] field_local;
}
void Field::evalDensity_sycl2() {

  sycl::queue q(sycl::default_selector_v);
  std::cout << " Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  double xmin = -10.0, xmax = 10.0;
  double ymin = -10.0, ymax = 10.0;
  double zmin = -5.0, zmax = 5.0;
  double delta = 0.25;
  vector<double> field;

  int npoints_x = int((xmax - xmin) / delta);
  int npoints_y = int((ymax - ymin) / delta);
  int npoints_z = int((zmax - zmin) / delta);
  const size_t nsize = npoints_x * npoints_y * npoints_z;
  int natm = wf.natm;
  int npri = wf.npri;
  int norb = wf.norb;
  double *field_local = new double[nsize];

  std::cout << " Points ( " << npoints_x << "," << npoints_y << "," << npoints_z
            << ")" << std::endl;
  std::cout << " TotalPoints : " << npoints_x * npoints_y * npoints_z
            << std::endl;

  double *coor = new double[3 * natm];
  for (int i = 0; i < natm; i++) {
    Rvector R(wf.atoms[i].getCoors());
    coor[3 * i] = R.get_x();
    coor[3 * i + 1] = R.get_y();
    coor[3 * i + 2] = R.get_z();
  }

  {
    sycl::buffer<int, 1> icnt_buff(wf.icntrs.data(), sycl::range<1>(npri));
    sycl::buffer<int, 1> vang_buff(wf.vang.data(), sycl::range<1>(3 * npri));
    sycl::buffer<double, 1> coor_buff(coor, sycl::range<1>(3 * natm));
    sycl::buffer<double, 1> eprim_buff(wf.depris.data(), sycl::range<1>(npri));
    sycl::buffer<double, 1> coef_buff(wf.dcoefs.data(),
                                      sycl::range<1>(npri * norb));
    sycl::buffer<double, 1> nocc_buff(wf.dnoccs.data(), sycl::range<1>(norb));
    sycl::buffer<double, 1> field_buff(field_local, sycl::range<1>(nsize));

    q.submit([&](sycl::handler &h) {
      auto field_acc = field_buff.get_access<sycl::access::mode::write>(h);
      auto icnt_acc = icnt_buff.get_access<sycl::access::mode::read>(h);
      auto vang_acc = vang_buff.get_access<sycl::access::mode::read>(h);
      auto coor_acc = coor_buff.get_access<sycl::access::mode::read>(h);
      auto eprim_acc = eprim_buff.get_access<sycl::access::mode::read>(h);
      auto coef_acc = coef_buff.get_access<sycl::access::mode::read>(h);
      auto nocc_acc = nocc_buff.get_access<sycl::access::mode::read>(h);

      h.parallel_for<class Field3>(
          sycl::range<3>(npoints_x, npoints_y, npoints_z),
          [=](sycl::id<3> idx) {
            double cart[3];
            int k = idx[2];
            int j = idx[1];
            int i = idx[0];
            int iglob = i * npoints_y * npoints_z + j * npoints_z + k;

            cart[0] = xmin + i * delta;
            cart[1] = ymin + j * delta;
            cart[2] = zmin + k * delta;

            const int *icnt_ptr =
                icnt_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            const int *vang_ptr =
                vang_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            const double *coor_ptr =
                coor_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            const double *eprim_ptr =
                eprim_acc.get_multi_ptr<sycl::access::decorated::no>()
                    .get_raw();
            const double *nocc_ptr =
                nocc_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
            const double *coef_ptr =
                coef_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

            field_acc[iglob] =
                DensitySYCL2(norb, npri, icnt_ptr, vang_ptr, cart, coor_ptr,
                             eprim_ptr, nocc_ptr, coef_ptr);
          });
    });
    q.wait();
  }

  for (int i = 0; i < nsize; i++)
    field.push_back(field_local[i]);

  dumpCube(xmin, ymin, zmin, delta, npoints_x, npoints_y, npoints_z, field,
           "densitySYCL2.cube");
  dumpXYZ("structure.xyz");

  delete[] coor;
  delete[] field_local;
}

void Field::spherical(std::string filename) {
  std::ofstream fout(filename);

  if (!fout.is_open())
    std::cerr << "Error: The file can be opened." << std::endl;

  int maxnpoints = 10000;
  int iter = 0;
  int np = 434;
  Lebedev Leb(np);
  std::vector<Rvector> rvecs = Leb.getRvecs();
  std::vector<double> weights = Leb.getWeights();
  double den = 100;
  double denx, deny, denz;
  double rmin = 0.0;
  double delta = 0.01;
  double suma = 0.;
  double r;
  std::vector<double> vden;
  std::vector<double> vr;
  std::vector<double> xden;
  std::vector<double> yden;
  std::vector<double> zden;
  while (iter < maxnpoints && den > 1.E-7) {
    r = rmin + iter * delta;
    suma = 0.;
    for (int i = 0; i < np; i++) {
      suma += weights[i] * Density(Lebedev::transform(rvecs[i], r));
    }
    denx = Density(Rvector(r, 0., 0.));
    deny = Density(Rvector(0., r, 0.));
    denz = Density(Rvector(0., 0., r));
    den = suma * 4. * M_PI;
    fout << std::setw(14) << std::setprecision(7) << std::fixed << r << ' ';
    fout << std::setw(14) << std::setprecision(7) << std::scientific << den
         << ' ';
    fout << std::setw(14) << std::setprecision(7) << std::scientific << denx
         << ' ';
    fout << std::setw(14) << std::setprecision(7) << std::scientific << deny
         << ' ';
    fout << std::setw(14) << std::setprecision(7) << std::scientific << denz
         << ' ';
    fout << std::endl;

    vden.push_back(den);
    vr.push_back(r);
    xden.push_back(denx);
    yden.push_back(deny);
    zden.push_back(denz);
    iter++;
  }

  fout.close();

  int last = vr.size() - 1;
  double integral = pow(vr[0], 2) * vden[0] + pow(vr[last], 2) * vden[last];
  double integralx = pow(vr[0], 2) * xden[0] + pow(vr[last], 2) * xden[last];
  double integraly = pow(vr[0], 2) * yden[0] + pow(vr[last], 2) * yden[last];
  double integralz = pow(vr[0], 2) * zden[0] + pow(vr[last], 2) * zden[last];

  for (int i = 1; i < vr.size() - 1; i++) {
    double r2 = 2. * vr[i] * vr[i];
    integral += r2 * vden[i];
    integralx += r2 * xden[i];
    integraly += r2 * yden[i];
    integralz += r2 * zden[i];
  }
  integral *= 0.5 * delta;
  integralx *= 0.5 * delta;
  integraly *= 0.5 * delta;
  integralz *= 0.5 * delta;

  std::cout << " Integral Atomica " << std::endl;
  std::cout << "Angular: " << std::setw(14) << std::setprecision(7)
            << std::fixed << integral << std::endl;
  std::cout << "Den  x : " << std::setw(14) << std::setprecision(7)
            << std::fixed << integralx << std::endl;
  std::cout << "Den  y : " << std::setw(14) << std::setprecision(7)
            << std::fixed << integraly << std::endl;
  std::cout << "Den  z : " << std::setw(14) << std::setprecision(7)
            << std::fixed << integralz << std::endl;
}
