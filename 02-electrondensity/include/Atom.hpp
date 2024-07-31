#ifndef _ATOM_H_
#define _ATOM_H_

#include <iostream>
#include <string>
#include <vector>

using std::string;

using namespace std;

class Atom {
   public:
    std::vector<double> coor;

   private:
    int zatom = -1;
    double mass = 0.1;
    string symb = "00";
    string setSymbol(int);

    int setAtomicNumberfromSymbol(string);
    double setAtomicMass(int);

   public:
    Atom(int, double, double, double);
    Atom(int, double *);
    Atom(int, std::vector<double>);
    ~Atom();
    //***********************************************
    Atom(string, double, double, double);
    Atom(string, double *);
    Atom(string, std::vector<double>);
    Atom(const char *, double, double, double);
    Atom(const char *, double *);
    Atom(const char *, std::vector<double>);
    //***********************************************
    Atom &operator=(const Atom &at);
    //***********************************************
    double getMass();
    string getSymbol();
    std::vector<double> getCoors();
    double get_x();
    double get_y();
    double get_z();
    int get_atnum();
    double get_charge();
    //***********************************************
    friend ostream &operator<<(ostream &o, const Atom &);
};

#endif