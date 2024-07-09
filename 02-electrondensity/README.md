# HandleWF

The following task is to provide alternatives for the follwing code for optimization for the SYCL kernel. 

We have to main parts for the computation being performed by SYCL in the `Field` class. The function to compute the *density* is in 
the following `void Field::evalDensity_sycl()` 

I outlined the code for the kernel and provide explanation and possible methods for imporvement. But before we get into the 
SYCL code and kernel, let's see if we can replace the `Rvector` class with the `std::vector` provided by STL of `C++`. 

`Atom.hpp` has been adjusted to use STL `std::vector` instead of Rvector.

### Declarting Buffers 

```cpp
  sycl::buffer<int, 1>   icnt_buff   (wf.icntrs.data(), sycl::range<1>(npri));
  sycl::buffer<int, 1>   vang_buff   (wf.vang.data()  , sycl::range<1>(3*npri));
  sycl::buffer<double, 1> coor_buff  (coor            , sycl::range<1>(3*natm));
  sycl::buffer<double, 1> eprim_buff (wf.depris.data(), sycl::range<1>(npri));
  sycl::buffer<double, 1> coef_buff  (wf.dcoefs.data(), sycl::range<1>(npri*norb));
  sycl::buffer<double, 1> nocc_buff  (wf.dnoccs.data(), sycl::range<1>(norb));
  sycl::buffer<double, 1> field_buff (field_local, sycl::range<1>(nsize));
```

### Queue submit

```cpp
q.submit([&](sycl::handler &h) {
  .....
}
```
### Accessor 

```cpp
    auto field_acc = field_buff.get_access<sycl::access::mode::write>(h);
    auto icnt_acc = icnt_buff.get_access<sycl::access::mode::read>(h);
    auto vang_acc = vang_buff.get_access<sycl::access::mode::read>(h);
    auto coor_acc = coor_buff.get_access<sycl::access::mode::read>(h);
    auto eprim_acc = eprim_buff.get_access<sycl::access::mode::read>(h);
    auto coef_acc = coef_buff.get_access<sycl::access::mode::read>(h);
    auto nocc_acc = nocc_buff.get_access<sycl::access::mode::read>(h);
```

### Parallel_for

```cpp
 h.parallel_for<class Field2>(sycl::range<1>(nsize), [=](sycl::id<1> idx){
      double cart[3];
      int k = (int) idx % npoints_z;
      int j = ((int) idx/npoints_z) % npoints_y;
      int i = (int) idx / (npoints_z * npoints_y);

      cart[0] = xmin + i * delta;
      cart[1] = ymin + j * delta;
      cart[2] = zmin + k * delta;

      const int *icnt_ptr = icnt_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
      const int *vang_ptr = vang_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
      const double *coor_ptr = coor_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
      const double *eprim_ptr = eprim_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
      const double *nocc_ptr = nocc_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
      const double *coef_ptr = coef_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

      field_acc[idx] = DensitySYCL2(norb, npri, icnt_ptr, vang_ptr, cart, coor_ptr, eprim_ptr, nocc_ptr, coef_ptr);
    });
```

### Modified access to pointers directly from accessors 

```cpp
h.parallel_for<class Field2>(sycl::range<1>(nsize), [=](sycl::id<1> idx){
    double cart[3];
    int k = (int) idx % npoints_z;
    int j = ((int) idx/npoints_z) % npoints_y;
    int i = (int) idx / (npoints_z * npoints_y);

    cart[0] = xmin + i * delta;
    cart[1] = ymin + j * delta;
    cart[2] = zmin + k * delta;

    // Direct access of pointers from accessors 
    field_acc[idx] = DensitySYCL2(
                        norb, npri, icnt_acc.get_pointer(),
                        vang_acc.get_pointer(), cart, coor_acc.get_pointer(),
                        eprim_acc.get_pointer(), nocc_acc.get_pointer(),
                        coef_acc.get_pointer());
  });
```

# Loop optimization

Loop unrolling is an optimization method used to enhance parallel processing and boost the efficiency of specific 
computational operations, especially when applied in hardware settings like FPGAs. 

Useful library `#include <boost/align/aligned_allocator.hpp>`

Maybe included `#include <boost/align/aligned_allocator.hpp>` to create a memory-aligned `std::vector`.
