
// CPU CODE
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

// this call function 1D or 3D
        double den = DensitySYCL2(wf.norb, wf.npri, wf.icntrs.data(),
                                  wf.vang.data(), r, coor, wf.depris.data(),
                                  wf.dnoccs.data(), wf.dcoefs.data());

        field.push_back(den);
      }
    }
  }

  dumpCube(xmin, ymin, zmin, delta, npoints_x, npoints_y, npoints_z, field,
           "densityCPU.cube");
//  dumpXYZ("structure.xyz");

  delete[] coor;
}

