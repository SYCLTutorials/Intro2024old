#include "Wavefunction.hpp"

Wavefunction::Wavefunction() {
    natm = 0;
    norb = 0;
    npri = 0;
}

Wavefunction::~Wavefunction() {}

void Wavefunction::loadWF(string fname) {
    int itmp;
    double tmp;
    double tx, ty, tz;
    std::vector<int> atomicNumbers;
    std::vector<std::vector<double>> atomicCoordinates;

    std::ifstream file(fname);
    if (file.is_open()) {
        file.clear();
        file.seekg(0, std::ios::beg);
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("<Number of Occupied Molecular Orbitals>") !=
                std::string::npos) {
                std::getline(file, line);
                norb = std::stoi(line);
            }
            if (line.find("<Number of Primitives>") != std::string::npos) {
                std::getline(file, line);
                npri = std::stoi(line);
            }
            if (line.find("<Primitive Centers>") != std::string::npos) {
                readVector(file, icntrs, "</Primitive Centers>");
                for (auto &i : icntrs) i -= 1;
            }
            if (line.find("<Primitive Types>") != std::string::npos) {
                readVector(file, itypes, "</Primitive Types>");
            }
            if (line.find("<Primitive Exponents>") != std::string::npos) {
                readVector(file, depris, "</Primitive Exponents>");
            }
            if (line.find("<Molecular Orbital Occupation Numbers>") !=
                std::string::npos) {
                readVector(file, dnoccs,
                           "</Molecular Orbital Occupation Numbers>");
            }
            if (line.find("<MO Number>") != std::string::npos) {
                std::getline(file, line);
                std::getline(file, line);

                if (line.find("</MO Number>") != std::string::npos) {
                    for (int j = 0; j < npri; j++) {
                        file >> tmp;
                        dcoefs.push_back(tmp);
                    }
                }
            }
            if (line.find(string("<Number of Nuclei>")) != std::string::npos) {
                std::getline(file, line);
                natm = std::stoi(line);
            }
            if (line.find(string("<Nuclear Cartesian Coordinates>")) !=
                std::string::npos) {
                for (int i = 0; i < natm; i++) {
                    std::getline(file, line);
                    sscanf(line.c_str(), "%lf %lf %lf", &tx, &ty, &tz);
                    // atomicCoordinates.push_back(Rvector(tx, ty, tz));
                    atomicCoordinates.push_back(
                        std::vector<double>{tx, ty, tz});
                }
            }
            if (line.find(string("<Atomic Numbers>")) != std::string::npos) {
                for (int i = 0; i < natm; i++) {
                    getline(file, line);
                    sscanf(line.c_str(), "%d", &itmp);
                    atomicNumbers.push_back(itmp);
                }
            }
        }
        setAngularVector();

        for (int i = 0; i < natm; i++)
            addAtom(Atom(atomicNumbers[i], atomicCoordinates[i]));
        file.close();
    } else {
        std::cerr << " Error to open file " << fname << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Wavefunction::setAngularVector() {

    static const vector<tuple<int, int, int>> typeMappings = {
        {0, 0, 0}, /* Type 0 */ {0, 0, 0}, /* Type 1 */ {1, 0, 0}, /* Type 2 */
        {0, 1, 0}, /* Type 3 */ {0, 0, 1}, /* Type 4 */ {2, 0, 0}, /* Type 5 */
        {0, 2, 0}, /* Type 6 */ {0, 0, 2}, /* Type 7 */ {1, 1, 0}, /* Type 8 */
        {1, 0, 1}, /* Type 9 */ {0, 1, 1}, /* Type 10 */ {3, 0, 0}, /* Type 11 */
        {0, 3, 0}, /* Type 12 */ {0, 0, 3}, /* Type 13 */ {2, 1, 0}, /* Type 14 */
        {2, 0, 1}, /* Type 15 */ {0, 2, 1}, /* Type 16 */ {1, 2, 0}, /* Type 17 */
        {1, 0, 2}, /* Type 18 */ {0, 1, 2}, /* Type 19 */ {1, 1, 1}, /* Type 20 */
        {4, 0, 0}, /* Type 21 */ {0, 4, 0}, /* Type 22 */ {0, 0, 4}, /* Type 23 */
        {3, 1, 0}, /* Type 24 */ {3, 0, 1}, /* Type 25 */ {1, 3, 0}, /* Type 26 */
        {0, 3, 1}, /* Type 27 */ {1, 0, 3}, /* Type 28 */ {0, 1, 3}, /* Type 29 */
        {2, 2, 0}, /* Type 30 */ {2, 0, 2}, /* Type 31 */ {0, 2, 2}, /* Type 32 */
        {2, 1, 1}, /* Type 33 */ {1, 2, 1}, /* Type 34 */ {1,1, 2}, /* Type 35 */
        {0, 0, 5}, /* Type 36 */ {0, 1, 4}, /* Type 37 */ {0, 2, 3}, /* Type 38 */
        {0, 3, 2}, /* Type 39 */ {0, 4, 1}, /* Type 40 */ {1, 5, 0}, /* Type 41 */
        {1, 0, 4}, /* Type 42 */ {1, 1, 3}, /* Type 43 */ {1, 2, 2}, /* Type 44 */
        {1, 3, 1}, /* Type 45 */ {1, 4, 0}, /* Type 46 */ {2, 0, 3}, /* Type 47 */
        {2, 1, 2}, /* Type 48 */ {2, 2, 1}, /* Type 49 */ {2, 3, 0}, /* Type 50 */
        {3, 0, 2}, /* Type 51 */ {3, 1, 1}, /* Type 52 */ {3, 2, 0}, /* Type 53 */
        {4, 0, 1}, /* Type 54 */ {4, 0, 1}, /* Type 55 */ {5, 0, 0}, /* Type 56 */
    };

    vang.resize(3 * npri);
        std::fill(vang.begin(), vang.end(), 0);

        for (int i = 0; i < npri; i++) {
            int j = 3 * i;
            int typeIndex = itypes[i];
            if (typeIndex < 0 || typeIndex >= typeMappings.size()) {
                std::cerr << "Type of primitive unsupported!!\n";
                exit(EXIT_FAILURE);
            }
            const auto& [v1, v2, v3] = typeMappings[typeIndex];
            vang[j]     = v1;
            vang[j + 1] = v2;
            vang[j + 2] = v3;
        }
}

void Wavefunction::addAtom(Atom a) { atoms.push_back(a); }

template <typename T>
void Wavefunction::readVector(std::ifstream &file, std::vector<T> &vector,
                              std::string endblock) {
    std::string line;
    while (std::getline(file, line) && line != endblock) {
        std::istringstream iss(line);
        T value;
        while (iss >> value) {
            vector.push_back(value);
        }
    }
}

void Wavefunction::setIntegerOutput() {
    std::cout << std::setw(6) << std::fixed << std::setprecision(0);
}

void Wavefunction::setScientificOutput() {
    std::cout << std::setw(13) << std::scientific << std::setprecision(6);
};

template <typename T>
void Wavefunction::printVector(const std::vector<T> &vector) {
    int ncount;
    int count = 0;
    if (std::is_integral<T>::value) {
        ncount = 10;
        for (const auto &val : vector) {
            setIntegerOutput();
            count++;
            std::cout << val << ' ';
            if (count == ncount) {
                count = 0;
                std::cout << std::endl;
            }
        }
        if (count != 0) std::cout << std::endl;
    } else {
        if (std::is_floating_point<T>::value) {
            ncount = 5;
            for (const auto &val : vector) {
                setScientificOutput();
                count++;
                std::cout << val << ' ';
                if (count == ncount) {
                    count = 0;
                    std::cout << std::endl;
                }
            }
            if (count != 0) std::cout << std::endl;
        } else {
            for (const auto &val : vector) {
                std::cout << val << std::endl;
            }
        }
    }
}

void Wavefunction::printCoefficients() {
    int count;

    for (int i = 0; i < norb; i++) {
        std::cout << " Coefficients for Orbital : " << i + 1 << std::endl;
        count = 0;
        for (int j = 0; j < npri; j++) {
            setScientificOutput();
            count++;
            std::cout << dcoefs[i * npri + j] << ' ';
            if (count == 5) {
                count = 0;
                std::cout << std::endl;
            }
        }
        if (count != 0) std::cout << std::endl;
    }
}

void Wavefunction::printWF() {
    setIntegerOutput();
    std::cout << " Number of Atoms      : " << natm << std::endl;
    setIntegerOutput();
    std::cout << " Number of primitives : " << npri << std::endl;
    setIntegerOutput();
    std::cout << " Number of mol. orb.  : " << norb << std::endl;

    std::cout << " Centers of primitives: " << std::endl;
    printVector(icntrs);
    std::cout << " Type of primitives   : " << std::endl;
    printVector(itypes);
    std::cout << " Angular moments      : " << std::endl;
    printVector(vang);

    std::cout << " Exponent of primitive: " << std::endl;
    printVector(depris);
    std::cout << " Occupation number    : " << std::endl;
    printVector(dnoccs);
    std::cout << " Coefficients         : " << std::endl;
    printCoefficients();
    std::cout << " Atoms                : " << std::endl;
    printVector(atoms);
}