#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include "helper_math.h"
#include "defines.h"


void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels,
		   const sycl::nd_item<3> &item_ct1) {
	uint blockId = sycl::mul24((int)item_ct1.get_group(1),
				   (int)item_ct1.get_group_range(2)) + item_ct1.get_group(2);
	uint i = sycl::mul24((int)blockId, (int)item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);

	if (voxelOccupied[i] && (i < numVoxels)) {
		compactedVoxelArray[voxelOccupiedScan[i]] = i;
	}

	if (voxelOccupied[i] && (i < numVoxels)) {
		printf("Compact voxel %u: compactedIndex=%u\n", i, voxelOccupiedScan[i]);
	}
}

extern "C" void launch_compactVoxels(sycl::range<3> grid, sycl::range<3> threads, uint *compactedVoxelArray,
				     uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels) {
	dpct::get_in_order_queue().parallel_for(sycl::nd_range<3>(grid * threads, threads),
						[=](sycl::nd_item<3> item_ct1) {
		compactVoxels(compactedVoxelArray, voxelOccupied, voxelOccupiedScan, numVoxels, item_ct1);
	});

	getLastCudaError("compactVoxels failed");
}
			
