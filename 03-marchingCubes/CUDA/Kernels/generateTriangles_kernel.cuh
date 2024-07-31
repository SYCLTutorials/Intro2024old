#ifndef _GENERATE_TRIANGLES_KERNEL_CUH_
#define _GENERATE_TRIANGLES_KERNEL_CUH_

#include <cuda_runtime.h>
#include "tables.h"


__global__ void generateTriangles(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
				  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize,
				  float isoValue, uint activeVoxels, uint maxVerts, cudaTextureObject_t triTex,
				  cudaTextureObject_t numsVertsTex);

#endif

