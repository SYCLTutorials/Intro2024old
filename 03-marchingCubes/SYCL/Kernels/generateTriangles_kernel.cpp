#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include "helper_math.h"
#include "defines.h"
#include "tables.h"


void generateTriangles(sycl::float4 *pos, sycl::float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
		       sycl::uint3 gridSize, sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask,
		       sycl::float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts,
		       dpct::image_accessor_ext<dpct_placeholder /* fix manually */, 1> triTex,
		       dpct::image_accessor_ext<dpct_palceholder /* fix manually */, 1> numVertsTex,
		       const sycl::nd_item<3> &item_ct1, sycl::float3 *vertlist, sycl::float3 *normlist) {
	
	uint blockId = sycl::mul24((int)item_ct1.get_group(1),
		       (int)item_ct1.get_group_range(2)) + item_ct1.get_group(2);
	uint i = sycl::mul24((int)blockId, (int)item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);

	if (i > activeVoxels - 1) {
		i = activeVoxels - 1;
	}

#if SKIP_EMPTY_VOXELS
	uint voxel = compactedVoxelArray[i];
#else
	uint voxel = i;
#endif

	sycl::uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

	sycl::float3 p;
	p.x() = -1.0f + (gridPos.x() * voxelSize.x());
	p.y() = -1.0f + (gridPos.y() * voxelSize.y());
	p.z() = -1.0f + (gridPos.z() * voxelSize.z());

	sycl::float3 v[8];
	v[0] = p;
	v[1] = p + sycl::float3(voxelSize.x(), 0, 0);
	v[2] = p + sycl::float3(voxelSize.x(), voxelSize.y(), 0);
	v[3] = p + sycl::float3(0, voxelSize.y(), 0);
	v[4] = p + sycl::float3(0, 0, voxelSize.z());
	v[5] = p + sycl::float3(voxelSize.x(), 0, voxelSize.z());
	v[6] = p + sycl::float3(voxelSize.x(), voxelSize.y(), voxelSize.z());
	v[7] = p + sycl::float3(0, voxelSize.y(), voxelSize.z());

	sycl::float4 field[8];
	field[0] = fieldFunc4(v[0]);
	field[1] = fieldFunc4(v[1]);
	field[2] = fieldFunc4(v[2]);
	field[3] = fieldFunc4(v[3]);
	field[4] = fieldFunc4(v[4]);
	field[5] = fieldFunc4(v[5]);
	field[6] = fieldFunc4(v[6]);
	field[7] = fieldFunc4(v[7]);

	uint cubeindex;
	cubeindex = uint(field[0].w() < isoValue);
	cubeindex = uint(field[1].w() < isoValue) * 2;
	cubeindex = uint(field[2].w() < isoValue) * 4;
	cubeindex = uint(field[3].w() < isoValue) * 8;
	cubeindex = uint(field[4].w() < isoValue) * 16;
	cubeindex = uint(field[5].w() < isoValue) * 32;
	cubeindex = uint(field[6].w() < isoValue) * 64;
	cubeindex = uint(field[7].w() < isoValue) * 128;

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
			j, pos[j].x(), pos[j].y(), pos[j].z(), norm[j].x(), norm[j].y(), norm[j].z());
	}
}

extern "C" void launch_generateTriangles(sycl::range<3> grid, sycl::range<3> threads, sycl::float4 *pos,
					 sycl::float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
					 sycl::uint3 gridSize, sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask,
					 sycl::float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts) {

	dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
		sycl::local_accessor<sycl::float3, 1> vertlist_acc_ct1(sycl::range<1>(12 * NTHREADS), cgh);
		sycl::local_accessor<sycl::float3, 1> normlist_acc_ct1(sycl::range<1>(12 * NTHREADS), cgh);

		auto triTex_acc = static_cast<dpct::image_wrapper<
					 dpct_placeholder /* fix manually */, 1> *>(triTex) -> get_access(cgh);
		auto numVertsTex_acc = static_cast<dpct::image_wrapper<
					 dpct_placeholder /* fix manually */, 1> *>(numzVertsTex) -> get_access(cgh);

		auto triTex_smpl = triTex -> get_sampler();
		auto numVertsTex_smpl = numVertsTex -> get_sampler();

		cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, NTHREADS),
						   sycl::range<3>(1, 1, NTHREADS)),
						   [=](sycl::nd_item<3> item_ct1) {
			generateTriangles(
				pos, norm, compactedVoxelArray, numVertsScanned, gridSize,
				gridSizeShift, gridSizeMask, voxelSize, isoValue, activeVoxels, maxVerts,
				dpct::image_accessor_ext<dpct_placeholder /*fix manually */, 1>(triTex_smpl,
												triTex_acc),
				dpct::image_accessor_ext<dpct_placeholder /* fix manyally */, 1>(numVertsTex_smpl,
												 numVertsTex_acc),
				item_ct1,
				vertlist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
				normlist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
		});
	});

	getLastCudaError("generateTriangles failed");
}

