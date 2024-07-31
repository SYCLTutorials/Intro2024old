#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include "helper_math.h"
#include "defines.h"
#include "tables.h"


void classifyVoxel(uint *voxelVerts, uint *voxelOccupied, uchar *volume, sycl::uint3 gridSize,
		   sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask, uint numVoxels,
		   sycl::float3 voxelSize, float isoValue,
		   dpct::image_accessor_ext<dpct_placeholder /* Fix this manually */, 1> numVertsTex,
		   dpct::image_accessor_ext<dpct_placeholder /* Fix this manually */, 1> volumeTex,
		   const sycl::nd_item<3> &item_ct1) {

	uint blockId = sycl::mul24((int)item_ct1.get_group(1),
		       (int)item_ct1.get_group_range(2)) + item_ct1.get_group(2);
	
	uint i = sycl::mul24((int)blockId, (int)item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);
	
	sycl::uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

	float field[8];
	field[0] = sampleVolume(volumeTex, volume, gridPos, gridSize);
	field[1] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 0), gridSize);
	field[2] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 0), gridSize);
	field[3] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 0), gridSize);
	field[4] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 0, 1), gridSize);
	field[5] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 1), gridSize);
	field[6] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 1), gridSize);
	field[7] = sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 1), gridSize);

	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
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

extern "C" void launch_classifyVoxel(sycl::range<3> grid, sycl::range<3> threads, uint *voxelVerts,
				     uint *voxelOccupied, uchar *volume, sycl::uint3 gridSize,
				     sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask, uint numVoxels,
				     sycl::float3 voxelSize, float isoValue) {

	dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
		auto numVertsTex_acc = static_cast<dpct::image_wrapper<
				dpct_placeholder /* Fix this manually */, 1> *>(numVertsTex) -> get_access(cgh);
		auto volumeTex_acc = static_cast<dpct::image_wrapper<
				dpct_placeholder /* Fi this manually */, 1> *>(volumeTex) -> get_access(cgh);

		auto numVertsTex_smpl = numVertsTex -> get_sampler();
		auto volumeTex_smpl = volumeTex -> get_sampler();

		cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads), [=](sycl::nd_item<3> item_ct1) {
			classifyVoxel(
				voxelVerts, voxelOccupied, volume, gridSize, gridSizeShift,
				gridSizeMask, numVoxels, voselSize, isoValue,
				dpct::image_accessor_ext<dpct_placeholder /* fix this manually */, 1>(
					numVertsTex_smpl, numVertsTex_acc),
				dpct::image_accessor_ext<dpct_placeholder /* fix this manually */, 1>(
					volumeTex_smpl, volumeTex_acc),
				item_ct1);
		});
	});
	
	getLastCudaError("classifyVoxel failed");
}

