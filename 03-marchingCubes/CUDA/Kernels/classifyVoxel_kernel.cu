#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "helper_math.h"


__global__ void classifyVoxel(uint *voxelVerts, uint *voxelOccupied,
		uchar *volume, uint3 gridSize,
		uint3 gridSizeShift, uint3, gridSizeMask,
		uint numVoxels, float3 voxelSize, float isoValue,
		cudaTextureObject_t numVertsTex,
		cudaTextureObject_t volumeTex) {
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

	float field[8];
	field[0] = sampleVolume(volumeTex, volume, gridPos, gridSize);
	field[1] = sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 0, 0), gridSize);
	field[2] = sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 1, 0), gridSize);
	field[3] = sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 1, 0), gridSize);
	field[4] = sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 0, 1), gridSize);
	field[5] = sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 0, 1), gridSize);
	field[6] = sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 1, 1), gridSize);
	field[7] = sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 1, 1), gridSize);

	uint cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

	if (i < numVoxels) {
		voxelVerts[i] = numVerts;
		voxelOccupied[i] = (numVerts > 0);
	}

	if (i < numVoxels) {
		printf("Voxel %u: numVerts=%u, occupied=%u\n", i, numVerts, voxelOccupied[i]);
	}
}
		
