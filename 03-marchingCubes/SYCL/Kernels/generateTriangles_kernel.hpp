#ifndef _GENERATE_TRIANGLES_KERNEL_HPP_
#define _GENERATE_TRIANGLES_KERNEL_HPP_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>


void generateTriangles(sycl::float4 *pos, sycl::float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
		       sycl::uint3 gridSize, sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask,
		       sycl::float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts,
		       dpct::image_accessor_ext<dpct_placeholder, 1> triTex,
		       dpct::image_accessor_ext<dpct_placeholder, 1> numVertsTex,
		       const sycl::nd_item<3> &item_ct1, sycl::float3 *vertlist, sycl::float3 *normlist);

extern "C" void launch_generateTriangles(sycl::range<3> grid, sycl::range<3> threads, sycl::float4 *pos,
					 sycl::float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
					 sycl::uint3 gridSize, sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask,
					 sycl::float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts);

#endif

