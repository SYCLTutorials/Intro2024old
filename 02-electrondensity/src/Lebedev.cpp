#include "Lebedev.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

Lebedev::Lebedev(int np) {
  npoints = getCorrectNPoint(np);
  switch (npoints) {
  case 6:
    Lebedev0006();
    break;
  case 14:
    Lebedev0014();
    break;
  case 26:
    Lebedev0026();
    break;
  case 38:
    Lebedev0038();
    break;
  case 50:
    Lebedev0050();
    break;
  case 74:
    Lebedev0074();
    break;
  case 86:
    Lebedev0086();
    break;
  case 110:
    Lebedev0110();
    break;
  case 146:
    Lebedev0146();
    break;
  case 170:
    Lebedev0170();
    break;
  case 194:
    Lebedev0194();
    break;
  case 230:
    Lebedev0230();
    break;
  case 266:
    Lebedev0266();
    break;
  case 302:
    Lebedev0302();
    break;
  case 350:
    Lebedev0350();
    break;
  case 434:
    Lebedev0434();
    break;
  }
}

int Lebedev::getCorrectNPoint(int np) {
  vector<int> values = {6,   14,  26,  38,  50,  74,  86,  110,
                        146, 170, 194, 230, 266, 302, 350, 434};
  int closest = values[0];
  int mindiff = abs(values[0] - np);

  for (int num : values) {
    int diff = abs(num - np);
    if (diff < mindiff) {
      mindiff = diff;
      closest = num;
    }
  }
  return closest;
}
std::vector<Rvector> &Lebedev::getRvecs() { return rvecs; }
std::vector<double> &Lebedev::getWeights() { return weights; }

void Lebedev::genGroup01(double va, double vb, double v) {
  rvecs.emplace_back(1., 0., 0.);
  weights.push_back(v);
  rvecs.emplace_back(-1., 0., 0.);
  weights.push_back(v);
  rvecs.emplace_back(0., 1., 0.);
  weights.push_back(v);
  rvecs.emplace_back(0., -1., 0.);
  weights.push_back(v);
  rvecs.emplace_back(0., 0., 1.);
  weights.push_back(v);
  rvecs.emplace_back(0., 0., -1.);
  weights.push_back(v);
}

void Lebedev::genGroup02(double va, double vb, double v) {
  double a = sqrt(0.5);

  rvecs.emplace_back(0., a, a);
  weights.push_back(v);
  rvecs.emplace_back(0., a, -a);
  weights.push_back(v);
  rvecs.emplace_back(0., -a, a);
  weights.push_back(v);
  rvecs.emplace_back(0., -a, -a);
  weights.push_back(v);
  rvecs.emplace_back(a, 0., a);
  weights.push_back(v);
  rvecs.emplace_back(a, 0., -a);
  weights.push_back(v);
  rvecs.emplace_back(-a, 0., a);
  weights.push_back(v);
  rvecs.emplace_back(-a, 0., -a);
  weights.push_back(v);
  rvecs.emplace_back(a, a, 0.);
  weights.push_back(v);
  rvecs.emplace_back(a, -a, 0.);
  weights.push_back(v);
  rvecs.emplace_back(-a, a, 0.);
  weights.push_back(v);
  rvecs.emplace_back(-a, -a, 0.);
  weights.push_back(v);
}

void Lebedev::genGroup03(double va, double vb, double v) {
  double a = 1. / sqrt(3.);

  rvecs.emplace_back(a, a, a);
  weights.push_back(v);
  rvecs.emplace_back(a, a, -a);
  weights.push_back(v);
  rvecs.emplace_back(a, -a, a);
  weights.push_back(v);
  rvecs.emplace_back(a, -a, -a);
  weights.push_back(v);
  rvecs.emplace_back(-a, a, a);
  weights.push_back(v);
  rvecs.emplace_back(-a, a, -a);
  weights.push_back(v);
  rvecs.emplace_back(-a, -a, a);
  weights.push_back(v);
  rvecs.emplace_back(-a, -a, -a);
  weights.push_back(v);
}

void Lebedev::genGroup04(double va, double vb, double v) {
  double a = va;
  double b = sqrt(1. - 2. * va * va);

  rvecs.emplace_back(a, a, b);
  weights.push_back(v);
  rvecs.emplace_back(-a, a, b);
  weights.push_back(v);
  rvecs.emplace_back(a, -a, b);
  weights.push_back(v);
  rvecs.emplace_back(-a, -a, b);
  weights.push_back(v);
  rvecs.emplace_back(a, a, -b);
  weights.push_back(v);
  rvecs.emplace_back(-a, a, -b);
  weights.push_back(v);
  rvecs.emplace_back(a, -a, -b);
  weights.push_back(v);
  rvecs.emplace_back(-a, -a, -b);
  weights.push_back(v);
  rvecs.emplace_back(a, b, a);
  weights.push_back(v);
  rvecs.emplace_back(-a, b, a);
  weights.push_back(v);
  rvecs.emplace_back(a, b, -a);
  weights.push_back(v);
  rvecs.emplace_back(-a, b, -a);
  weights.push_back(v);
  rvecs.emplace_back(a, -b, a);
  weights.push_back(v);
  rvecs.emplace_back(-a, -b, a);
  weights.push_back(v);
  rvecs.emplace_back(a, -b, -a);
  weights.push_back(v);
  rvecs.emplace_back(-a, -b, -a);
  weights.push_back(v);
  rvecs.emplace_back(b, a, a);
  weights.push_back(v);
  rvecs.emplace_back(b, -a, a);
  weights.push_back(v);
  rvecs.emplace_back(b, a, -a);
  weights.push_back(v);
  rvecs.emplace_back(b, -a, -a);
  weights.push_back(v);
  rvecs.emplace_back(-b, a, a);
  weights.push_back(v);
  rvecs.emplace_back(-b, -a, a);
  weights.push_back(v);
  rvecs.emplace_back(-b, a, -a);
  weights.push_back(v);
  rvecs.emplace_back(-b, -a, -a);
  weights.push_back(v);
}

void Lebedev::genGroup05(double va, double vb, double v) {
  double a = va;
  double b = sqrt(1. - va * va);

  rvecs.emplace_back(a, b, 0);
  weights.push_back(v);
  rvecs.emplace_back(a, -b, 0);
  weights.push_back(v);
  rvecs.emplace_back(-a, b, 0);
  weights.push_back(v);
  rvecs.emplace_back(-a, -b, 0);
  weights.push_back(v);
  rvecs.emplace_back(b, a, 0);
  weights.push_back(v);
  rvecs.emplace_back(-b, a, 0);
  weights.push_back(v);
  rvecs.emplace_back(b, -a, 0);
  weights.push_back(v);
  rvecs.emplace_back(-b, -a, 0);
  weights.push_back(v);
  rvecs.emplace_back(a, 0, b);
  weights.push_back(v);
  rvecs.emplace_back(a, 0, -b);
  weights.push_back(v);
  rvecs.emplace_back(-a, 0, b);
  weights.push_back(v);
  rvecs.emplace_back(-a, 0, -b);
  weights.push_back(v);
  rvecs.emplace_back(b, 0, a);
  weights.push_back(v);
  rvecs.emplace_back(-b, 0, a);
  weights.push_back(v);
  rvecs.emplace_back(b, 0, -a);
  weights.push_back(v);
  rvecs.emplace_back(-b, 0, -a);
  weights.push_back(v);
  rvecs.emplace_back(0, a, b);
  weights.push_back(v);
  rvecs.emplace_back(0, a, -b);
  weights.push_back(v);
  rvecs.emplace_back(0, -a, b);
  weights.push_back(v);
  rvecs.emplace_back(0, -a, -b);
  weights.push_back(v);
  rvecs.emplace_back(0, b, a);
  weights.push_back(v);
  rvecs.emplace_back(0, -b, a);
  weights.push_back(v);
  rvecs.emplace_back(0, b, -a);
  weights.push_back(v);
  rvecs.emplace_back(0, -b, -a);
  weights.push_back(v);
}
void Lebedev::genGroup06(double va, double vb, double v) {
  double a = va;
  double b = vb;
  double c = sqrt(1. - va * va - vb * vb);

  rvecs.emplace_back(a, b, c);
  weights.push_back(v);
  rvecs.emplace_back(a, -b, c);
  weights.push_back(v);
  rvecs.emplace_back(-a, b, c);
  weights.push_back(v);
  rvecs.emplace_back(-a, -b, c);
  weights.push_back(v);
  rvecs.emplace_back(b, a, c);
  weights.push_back(v);
  rvecs.emplace_back(-b, a, c);
  weights.push_back(v);
  rvecs.emplace_back(b, -a, c);
  weights.push_back(v);
  rvecs.emplace_back(-b, -a, c);
  weights.push_back(v);
  rvecs.emplace_back(a, c, b);
  weights.push_back(v);
  rvecs.emplace_back(a, c, -b);
  weights.push_back(v);
  rvecs.emplace_back(-a, c, b);
  weights.push_back(v);
  rvecs.emplace_back(-a, c, -b);
  weights.push_back(v);
  rvecs.emplace_back(b, c, a);
  weights.push_back(v);
  rvecs.emplace_back(-b, c, a);
  weights.push_back(v);
  rvecs.emplace_back(b, c, -a);
  weights.push_back(v);
  rvecs.emplace_back(-b, c, -a);
  weights.push_back(v);
  rvecs.emplace_back(c, a, b);
  weights.push_back(v);
  rvecs.emplace_back(c, a, -b);
  weights.push_back(v);
  rvecs.emplace_back(c, -a, b);
  weights.push_back(v);
  rvecs.emplace_back(c, -a, -b);
  weights.push_back(v);
  rvecs.emplace_back(c, b, a);
  weights.push_back(v);
  rvecs.emplace_back(c, -b, a);
  weights.push_back(v);
  rvecs.emplace_back(c, b, -a);
  weights.push_back(v);
  rvecs.emplace_back(c, -b, -a);
  weights.push_back(v);
  rvecs.emplace_back(a, b, -c);
  weights.push_back(v);
  rvecs.emplace_back(a, -b, -c);
  weights.push_back(v);
  rvecs.emplace_back(-a, b, -c);
  weights.push_back(v);
  rvecs.emplace_back(-a, -b, -c);
  weights.push_back(v);
  rvecs.emplace_back(b, a, -c);
  weights.push_back(v);
  rvecs.emplace_back(-b, a, -c);
  weights.push_back(v);
  rvecs.emplace_back(b, -a, -c);
  weights.push_back(v);
  rvecs.emplace_back(-b, -a, -c);
  weights.push_back(v);
  rvecs.emplace_back(a, -c, b);
  weights.push_back(v);
  rvecs.emplace_back(a, -c, -b);
  weights.push_back(v);
  rvecs.emplace_back(-a, -c, b);
  weights.push_back(v);
  rvecs.emplace_back(-a, -c, -b);
  weights.push_back(v);
  rvecs.emplace_back(b, -c, a);
  weights.push_back(v);
  rvecs.emplace_back(-b, -c, a);
  weights.push_back(v);
  rvecs.emplace_back(b, -c, -a);
  weights.push_back(v);
  rvecs.emplace_back(-b, -c, -a);
  weights.push_back(v);
  rvecs.emplace_back(-c, a, b);
  weights.push_back(v);
  rvecs.emplace_back(-c, a, -b);
  weights.push_back(v);
  rvecs.emplace_back(-c, -a, b);
  weights.push_back(v);
  rvecs.emplace_back(-c, -a, -b);
  weights.push_back(v);
  rvecs.emplace_back(-c, b, a);
  weights.push_back(v);
  rvecs.emplace_back(-c, -b, a);
  weights.push_back(v);
  rvecs.emplace_back(-c, b, -a);
  weights.push_back(v);
  rvecs.emplace_back(-c, -b, -a);
  weights.push_back(v);
}

void Lebedev::Lebedev0006() {
  double a = 0., b = 0., v = 0.;
  v = 0.1666666666666667;
  genGroup01(a, b, v);
}

void Lebedev::Lebedev0014() {
  double a = 0., b = 0., v = 0.;
  v = 0.6666666666666667E-1;
  genGroup01(a, b, v);
  v = 0.7500000000000000E-1;
  genGroup03(a, b, v);
}

void Lebedev::Lebedev0026() {
  double a = 0., b = 0., v = 0.;
  v = 0.4761904761904762E-1;
  genGroup01(a, b, v);
  v = 0.3809523809523810E-1;
  genGroup02(a, b, v);
  v = 0.3214285714285714E-1;
  genGroup03(a, b, v);
}

void Lebedev::Lebedev0038() {
  double a = 0., b = 0., v = 0.;
  v = 0.9523809523809524E-2;
  genGroup01(a, b, v);
  v = 0.3214285714285714E-1;
  genGroup03(a, b, v);
  a = 0.4597008433809831;
  v = 0.2857142857142857E-1;
  genGroup05(a, b, v);
}

void Lebedev::Lebedev0050() {
  double a = 0., b = 0., v = 0.;
  v = 0.1269841269841270E-1;
  genGroup01(a, b, v);
  v = 0.2257495590828924E-1;
  genGroup02(a, b, v);
  v = 0.2109375000000000E-1;
  genGroup03(a, b, v);
  a = 0.3015113445777636;
  v = 0.2017333553791887E-1;
  genGroup04(a, b, v);
}

void Lebedev::Lebedev0074() {
  double a = 0., b = 0., v = 0.;
  v = 0.5130671797338464E-3;
  genGroup01(a, b, v);
  v = 0.1660406956574204E-1;
  genGroup02(a, b, v);
  v = -0.2958603896103896E-1;
  genGroup03(a, b, v);
  a = 0.4803844614152614;
  v = 0.2657620708215946E-1;
  genGroup04(a, b, v);
  a = 0.3207726489807764;
  v = 0.1652217099371571E-1;
  genGroup05(a, b, v);
}

void Lebedev::Lebedev0086() {
  double a = 0., b = 0., v = 0.;
  v = 0.1154401154401154E-1;
  genGroup01(a, b, v);
  v = 0.1194390908585628E-1;
  genGroup03(a, b, v);
  a = 0.3696028464541502;
  v = 0.1111055571060340E-1;
  genGroup04(a, b, v);
  a = 0.6943540066026664;
  v = 0.1187650129453714E-1;
  genGroup04(a, b, v);
  a = 0.3742430390903412;
  v = 0.1181230374690448E-1;
  genGroup05(a, b, v);
}

void Lebedev::Lebedev0110() {
  double a = 0., b = 0., v = 0.;
  v = 0.3828270494937162E-2;
  genGroup01(a, b, v);
  v = 0.9793737512487512E-2;
  genGroup03(a, b, v);
  a = 0.1851156353447362;
  v = 0.8211737283191111E-2;
  genGroup04(a, b, v);
  a = 0.6904210483822922;
  v = 0.9942814891178103E-2;
  genGroup04(a, b, v);
  a = 0.3956894730559419;
  v = 0.9595471336070963E-2;
  genGroup04(a, b, v);
  a = 0.4783690288121502;
  v = 0.9694996361663028E-2;
  genGroup05(a, b, v);
}

void Lebedev::Lebedev0146() {
  double a = 0., b = 0., v = 0.;
  v = 0.5996313688621381E-3;
  genGroup01(a, b, v);
  v = 0.7372999718620756E-2;
  genGroup02(a, b, v);
  v = 0.7210515360144488E-2;
  genGroup03(a, b, v);
  a = 0.6764410400114264;
  v = 0.7116355493117555E-2;
  genGroup04(a, b, v);
  a = 0.4174961227965453;
  v = 0.6753829486314477E-2;
  genGroup04(a, b, v);
  a = 0.1574676672039082;
  v = 0.7574394159054034E-2;
  genGroup04(a, b, v);
  a = 0.1403553811713183;
  b = 0.4493328323269557;
  v = 0.6991087353303262E-2;
  genGroup06(a, b, v);
}

void Lebedev::Lebedev0170() {
  double a = 0., b = 0., v = 0.;
  v = 0.5544842902037365E-2;
  genGroup01(a, b, v);
  v = 0.6071332770670752E-2;
  genGroup02(a, b, v);
  v = 0.6383674773515093E-2;
  genGroup03(a, b, v);
  a = 0.2551252621114134;
  v = 0.5183387587747790E-2;
  genGroup04(a, b, v);
  a = 0.6743601460362766;
  v = 0.6317929009813725E-2;
  genGroup04(a, b, v);
  a = 0.4318910696719410;
  v = 0.6201670006589077E-2;
  genGroup04(a, b, v);
  a = 0.2613931360335988;
  v = 0.5477143385137348E-2;
  genGroup05(a, b, v);
  a = 0.4990453161796037;
  b = 0.1446630744325115;
  v = 0.5968383987681156E-2;
  genGroup06(a, b, v);
}

void Lebedev::Lebedev0194() {
  double a = 0., b = 0., v = 0.;
  v = 0.1782340447244611E-2;
  genGroup01(a, b, v);
  v = 0.5716905949977102E-2;
  genGroup02(a, b, v);
  v = 0.5573383178848738E-2;
  genGroup03(a, b, v);
  a = 0.6712973442695226;
  v = 0.5608704082587997E-2;
  genGroup04(a, b, v);
  a = 0.2892465627575439;
  v = 0.5158237711805383E-2;
  genGroup04(a, b, v);
  a = 0.4446933178717437;
  v = 0.5518771467273614E-2;
  genGroup04(a, b, v);
  a = 0.1299335447650067;
  v = 0.4106777028169394E-2;
  genGroup04(a, b, v);
  a = 0.3457702197611283;
  v = 0.5051846064614808E-2;
  genGroup05(a, b, v);
  a = 0.1590417105383530;
  b = 0.8360360154824589;
  v = 0.5530248916233094E-2;
  genGroup06(a, b, v);
}
void Lebedev::Lebedev0230() {
  double a = 0., b = 0., v = 0.;
  v = -0.5522639919727325E-1;
  genGroup01(a, b, v);
  v = 0.4450274607445226E-2;
  genGroup03(a, b, v);
  a = 0.4492044687397611;
  v = 0.4496841067921404E-2;
  genGroup04(a, b, v);
  a = 0.2520419490210201;
  v = 0.5049153450478750E-2;
  genGroup04(a, b, v);
  a = 0.6981906658447242;
  v = 0.3976408018051883E-2;
  genGroup04(a, b, v);
  a = 0.6587405243460960;
  v = 0.4401400650381014E-2;
  genGroup04(a, b, v);
  a = 0.4038544050097660E-1;
  v = 0.1724544350544401E-1;
  genGroup04(a, b, v);
  a = 0.5823842309715585;
  v = 0.4231083095357343E-2;
  genGroup05(a, b, v);
  a = 0.3545877390518688;
  v = 0.5198069864064399E-2;
  genGroup05(a, b, v);
  a = 0.2272181808998187;
  b = 0.4864661535886647;
  v = 0.4695720972568883E-2;
  genGroup06(a, b, v);
}

void Lebedev::Lebedev0266() {
  double a = 0., b = 0., v = 0.;
  v = -0.1313769127326952E-2;
  genGroup01(a, b, v);
  v = -0.2522728704859336E-2;
  genGroup02(a, b, v);
  v = 0.4186853881700583E-2;
  genGroup03(a, b, v);
  a = 0.7039373391585475;
  v = 0.5315167977810885E-2;
  genGroup04(a, b, v);
  a = 0.1012526248572414;
  v = 0.4047142377086219E-2;
  genGroup04(a, b, v);
  a = 0.4647448726420539;
  v = 0.4112482394406990E-2;
  genGroup04(a, b, v);
  a = 0.3277420654971629;
  v = 0.3595584899758782E-2;
  genGroup04(a, b, v);
  a = 0.6620338663699974;
  v = 0.4256131351428158E-2;
  genGroup04(a, b, v);
  a = 0.8506508083520399;
  v = 0.4229582700647240E-2;
  genGroup05(a, b, v);
  a = 0.3233484542692899;
  b = 0.1153112011009701;
  v = 0.4080914225780505E-2;
  genGroup06(a, b, v);
  a = 0.2314790158712601;
  b = 0.5244939240922365;
  v = 0.4071467593830964E-2;
  genGroup06(a, b, v);
}

void Lebedev::Lebedev0302() {
  double a = 0., b = 0., v = 0.;
  v = 0.8545911725128148E-3;
  genGroup01(a, b, v);
  v = 0.3599119285025571E-2;
  genGroup03(a, b, v);
  a = 0.3515640345570105;
  v = 0.3449788424305883E-2;
  genGroup04(a, b, v);
  a = 0.6566329410219612;
  v = 0.3604822601419882E-2;
  genGroup04(a, b, v);
  a = 0.4729054132581005;
  v = 0.3576729661743367E-2;
  genGroup04(a, b, v);
  a = 0.9618308522614784E-1;
  v = 0.2352101413689164E-2;
  genGroup04(a, b, v);
  a = 0.2219645236294178;
  v = 0.3108953122413675E-2;
  genGroup04(a, b, v);
  a = 0.7011766416089545;
  v = 0.3650045807677255E-2;
  genGroup04(a, b, v);
  a = 0.2644152887060663;
  v = 0.2982344963171804E-2;
  genGroup05(a, b, v);
  a = 0.5718955891878961;
  v = 0.3600820932216460E-2;
  genGroup05(a, b, v);
  a = 0.2510034751770465;
  b = 0.8000727494073952;
  v = 0.3571540554273387E-2;
  genGroup06(a, b, v);
  a = 0.1233548532583327;
  b = 0.4127724083168531;
  v = 0.3392312205006170E-2;
  genGroup06(a, b, v);
}

void Lebedev::Lebedev0350() {
  double a = 0., b = 0., v = 0.;
  v = 0.3006796749453936E-2;
  genGroup01(a, b, v);
  v = 0.3050627745650771E-2;
  genGroup03(a, b, v);
  a = 0.7068965463912316;
  v = 0.1621104600288991E-2;
  genGroup04(a, b, v);
  a = 0.4794682625712025;
  v = 0.3005701484901752E-2;
  genGroup04(a, b, v);
  a = 0.1927533154878019;
  v = 0.2990992529653774E-2;
  genGroup04(a, b, v);
  a = 0.6930357961327123;
  v = 0.2982170644107595E-2;
  genGroup04(a, b, v);
  a = 0.3608302115520091;
  v = 0.2721564237310992E-2;
  genGroup04(a, b, v);
  a = 0.6498486161496169;
  v = 0.3033513795811141E-2;
  genGroup04(a, b, v);
  a = 0.1932945013230339;
  v = 0.3007949555218533E-2;
  genGroup05(a, b, v);
  a = 0.3800494919899303;
  v = 0.2881964603055307E-2;
  genGroup05(a, b, v);
  a = 0.2899558825499574;
  b = 0.7934537856582316;
  v = 0.2958357626535696E-2;
  genGroup06(a, b, v);
  a = 0.9684121455103957E-1;
  b = 0.8280801506686862;
  v = 0.3036020026407088E-2;
  genGroup06(a, b, v);
  a = 0.1833434647041659;
  b = 0.9074658265305127;
  v = 0.2832187403926303E-2;
  genGroup06(a, b, v);
}

void Lebedev::Lebedev0434() {
  double a = 0., b = 0., v = 0.;
  v = 0.5265897968224436E-3;
  genGroup01(a, b, v);
  v = 0.2548219972002607E-2;
  genGroup02(a, b, v);
  v = 0.2512317418927307E-2;
  genGroup03(a, b, v);
  a = 0.6909346307509111;
  v = 0.2530403801186355E-2;
  genGroup04(a, b, v);
  a = 0.1774836054609158;
  v = 0.2014279020918528E-2;
  genGroup04(a, b, v);
  a = 0.4914342637784746;
  v = 0.2501725168402936E-2;
  genGroup04(a, b, v);
  a = 0.6456664707424256;
  v = 0.2513267174597564E-2;
  genGroup04(a, b, v);
  a = 0.2861289010307638;
  v = 0.2302694782227416E-2;
  genGroup04(a, b, v);
  a = 0.7568084367178018E-1;
  v = 0.1462495621594614E-2;
  genGroup04(a, b, v);
  a = 0.3927259763368002;
  v = 0.2445373437312980E-2;
  genGroup04(a, b, v);
  a = 0.8818132877794288;
  v = 0.2417442375638981E-2;
  genGroup05(a, b, v);
  a = 0.9776428111182649;
  v = 0.1910951282179532E-2;
  genGroup05(a, b, v);
  a = 0.2054823696403044;
  b = 0.8689460322872412;
  v = 0.2416930044324775E-2;
  genGroup06(a, b, v);
  a = 0.5905157048925271;
  b = 0.7999278543857286;
  v = 0.2512236854563495E-2;
  genGroup06(a, b, v);
  a = 0.5550152361076807;
  b = 0.7717462626915901;
  v = 0.2496644054553086E-2;
  genGroup06(a, b, v);
  a = 0.9371809858553722;
  b = 0.3344363145343455;
  v = 0.2236607760437849E-2;
  genGroup06(a, b, v);
}

Rvector Lebedev::transform(Rvector rvec, double rnew) {
  double x = rvec.get_x();
  double y = rvec.get_y();
  double z = rvec.get_z();

  double theta, phi;
  theta = acos(z);
  phi = atan2(y, x);
  x = rnew * sin(theta) * cos(phi);
  y = rnew * sin(theta) * sin(phi);
  z = rnew * cos(theta);

  return Rvector(x, y, z);
}