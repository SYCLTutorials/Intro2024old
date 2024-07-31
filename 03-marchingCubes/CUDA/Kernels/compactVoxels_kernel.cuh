#ifndef _COMPACT_VOXELS_KERNEL_CUH_
#define _COMPACT_VOXELS_KERNEL_CUH_

#include <cuda_runtime.h>


__global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied,
			      uint *voxelOccupiedScan, uint numVoxels);

#endif

