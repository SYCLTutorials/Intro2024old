#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h"


__global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied,
			      uint *voxelOccupiedScan, uint numVoxels) {
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (voxelOccupied[i] && (i < numVoxels)) {
		compactedVoxelArray[voxelOccupiedScan[i]] = i;
	}

	if (voxelOccupied[i] && (i < numVoxels)) {
		printf("Compact voxel %u: compactedIndex=%u\n", i , voxelOccupiedScan[i]);
	}
}

