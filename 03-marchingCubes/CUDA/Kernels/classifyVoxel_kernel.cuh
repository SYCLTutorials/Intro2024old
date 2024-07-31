#ifndef _CLASSIFY_VOXEL_KERNEL_CUH_
#define _CLASSIFY_VOXEL_KERNEL_CUH_

#include <cuda_runtime.h>


__global__ void classifyVoxel(uint *voxelVerts, uint *voxelOccupuied,
			      uchar *volume, uint3 gridSize,
			      uint3 gridSizeShift, uint3 gridSizeMask,
			      uint numVoxels, float3 voxelSize, float isoValue,
			      cudaTextureObject_t numVertsTex, cudaTextureObject_t volumeTex);

#endif

