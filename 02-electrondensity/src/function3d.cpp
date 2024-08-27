#include "setGrid.hpp"

void Field::evalDensity_sycl2() {

  sycl::queue q(sycl::default_selector_v);
  std::cout << " Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  int npoints_x, npoints_y, npoints_z;
  double xmin, ymin, zmin, delta;
  vector<double> field;
  size_t nsize;
  configGrid(&npoints_x, &npoints_y, &npoints_z, &delta, &xmin, &ymin, &zmin, &nsize);

  int natm = wf.natm;
  int npri = wf.npri;
  int norb = wf.norb;
  double *field_local = new double[nsize];

  std::cout << " Points ( " << npoints_x << "," << npoints_y << "," << npoints_z
            << ")" << std::endl;
  std::cout << " TotalPoints : " << nsize
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

// 3D index
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
 // dumpXYZ("structure.xyz");

  delete[] coor;
  delete[] field_local;
}