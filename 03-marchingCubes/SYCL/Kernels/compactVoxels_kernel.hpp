#ifndef _COMPACT_VOXELS_KERNEL_HPP_
#define _COMPACT_VOXELS_KERNEL_HPP_

#include <sycl/sycl.hpp>
#include <dcpt/dcpt.hpp>


void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan,
		   uint numVoxels, const sycl::nd_item<3> &item_ct1);

extern "C" void launch_compactVoxels(sycl::range<3> grid, sycl::range<3> threads, uint *compactedVoxelArray,
				     uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels);

#endif

