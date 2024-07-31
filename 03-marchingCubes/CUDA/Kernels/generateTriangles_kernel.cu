#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "tables.h"


__global__ void generatetriangles(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
				  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize,
				  float isoValue, uint activeVoxels, uint maxVerts, cudaTextureObject_t triTex,
				  cudaTextureObject_t numVertsTex) {
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (i > activeVoxels - 1) {
		i = activeVoxels - 1;
	}

	uint voxel = compactedVoxelArray[i];
	uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

	float3 p;
	p.x = -1.0f + (gridPos.x * voxelSize.x);
	p.y = -1.0f + (gridPos.y * voxelSize.y);
	p.z = -1.0f + (gridPos.z * voxelSize.z);

	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

	float4 field[8];
	field[0] = fieldFunc4(v[0]);
	field[1] = fieldFunc4(v[1]);
	field[2] = fieldFunc4(v[2]);
	field[3] = fieldFunc4(v[3]);
	field[4] = fieldFunc4(v[4]);
	field[5] = fieldFunc4(v[5]);
	field[6] = fieldFunc4(v[6]);
	field[7] = fieldFunc4(v[7]);

	uint cubeindex;
	cubeindex = uint(field[0].w < isoValue);
	cubeindex += uint(field[1].w < isoValue) * 2;
	cubeindex += uint(field[2].w < isoValue) * 4;
	cubeindex += uint(field[3].w < isoValue) * 8;
	cubeindex += uint(field[4].w < isoValue) * 16;
	cubeindex += uint(field[5].w < isoValue) * 32;
	cubeindex += uint(field[6].w < isoValue) * 64;
	cubeindex += uint(field[7].w < isoValue) * 128;

	float3 vertlist[12];
	float3 normlist[12];

	vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertlist[0], normlist[0]);
	vertexInterp2(isoValue, v[1], v[2], field[1], field[2], vertlist[1], normlist[1]);
	vertexInterp2(isoValue, v[2], v[3], field[2], field[3], vertlist[2], normlist[2]);
	vertexInterp2(isoValue, v[3], v[0], field[3], field[0], vertlist[3], normlist[3]);

	vertexInterp2(isoValue, v[4], v[5], field[4], field[5], vertlist[4], normlist[4]);
	vertexInterp2(isoValue, v[5], v[6], field[5], field[6], vertlist[5], normlist[5]);
	vertexInterp2(isoValue, v[6], v[7], field[6], field[7], vertlist[6], normlist[6]);
	vertexInterp2(isoValue, v[7], v[4], field[7], field[4], vertlist[7], normlist[7]);

	vertexInterp2(isoValue, v[0], v[4], field[0], field[4], vertlist[8], normlist[8]);
	vertexInterp2(isoValue, v[1], v[5], field[1], field[5], vertlist[9], normlist[9]);
	vertexInterp2(isoValue, v[2], v[6], field[2], field[6], vertlist[10], normlist[10]);
	vertexInterp2(isoValue, v[3], v[7], field[3], field[7], vertlist[11], normlist[11]);

	uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

	for (int i = 0; i < numVerts; i++) {
		uint edge = tex1Dfetch<uint>(triTex, cubeindex * 16 + i);
		uint index = numVertsScanned[voxel] + i;

		if (index < maxVerts) {
			pos[index] = make_float4(vertlist[edge], 1.0f);
			norm[index] = make_float4(normlist[edge], 0.0f);
		}
	}

	for (int j = 0; j < numVerts; j++) {
		printf("Triangle vertex %u: pos=(%f, %f, %f), norm=(%f, %f, %f)\n",
			j, pos[j].x, pos[j].y, pos[j].z, norm[j].x, norm[j].y, norm[j].z);
	}
}

