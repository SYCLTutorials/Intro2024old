# Electron density
Electron density is vital in theoretical and computational chemistry and materials science. It is an "observable" quantity, according to the quantum mechanical definition, so we can measure it experimentally (X-Ray, neutron diffraction), unlike the wave-function or the orbitals. Regardless of the number of particles, it is always a real function of three variables (x, y, z), whereas the wave-function of N particles is a function of 4 N variables (three for spatial coordinates and one for the spin).

According to the theorems of Hohenberg and Kohn, the electron density contains all the information of the system (molecular or solid) in its basal states because, in principle, by knowing the electron density, we know all the properties of the system.  The Hohenberg-Kohn theorems are the basis of the Density Functional Theory (DFT); today, DFT has become the workhorse of modern computational chemistry and materials science.
Electron density analysis *per se* provides a variety of knowledge, such as chemical bond analysis, reactivity and molecular properties like electrostatic potential.

This module will present a way to evaluate electron density (a 3D scalar field) using SYCL.

## Mathematical grounding

The electron density, when we are using localized basis set in molecular systems, can be  written in the following form as the sum of $N$ squared molecular orbitals { $\phi_i$ } multiplied by a value called the occupation number ($\omega_i$).

$$\rho(\vec{r}) = \sum_i^N \omega_i \left| \phi_i(\vec{r}) \right|^2$$

While the orbitals are  determined from a linear combination of $K$ primitive functions, Cartesian Gaussian in this case, centered on the different nuclei ($R_\mu$) of the molecule. The coefficients ($c_{i\mu}$) are obtained from computational chemistry methods.

$$\phi_i(\vec{r}) = \sum_\mu^K c_{i \mu} \; g\left(\vec{r}; \alpha_\mu, \vec{R}_\mu, \vec{l}_\mu\right)$$

Finally, Gaussian functions are 3-dimensional functions, as we mentioned above, they are centered on atoms that form the molecular system ($\vec{R} = (X, Y, Z)$) and whose exponents ($\alpha$) have been  previously optimized. Each Gaussian function is characterized by its center, the exponent, and the product of polynomials  in each Cartesian coordinate.

$$g\left(\vec{r}; \alpha, R, \vec{l}\right) = (x - X)^{l_x}(y - Y)^{l_y}(z - Z)^{l_z} \; e^{-\alpha|\vec{r}-\vec{R}|^2}$$
