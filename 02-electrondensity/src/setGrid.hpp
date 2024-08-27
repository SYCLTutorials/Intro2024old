#ifndef _SETGRID_HPP_
#define _SETGRID_HPP_

#include <cmath>

void configGrid(int  *npx, int *npy, int *npz, double *delta,
                double *xmin, double *ymin, double *zmin,
                size_t *nsize){

   (*xmin) = -10.0;
   (*ymin) = -10.0;
   (*zmin) = -10.0;

   (*delta) = 0.25;

   (*npx) = fabs(2.*(*xmin)) / (*delta);
   (*npy) = fabs(2.*(*ymin)) / (*delta);
   (*npz) = fabs(2.*(*zmin)) / (*delta);

   (*nsize) = (*npx) * (*npy) * (*npz);

 }

#endif
