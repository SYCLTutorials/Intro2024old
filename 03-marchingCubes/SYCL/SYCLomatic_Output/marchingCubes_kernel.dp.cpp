/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _MARCHING_CUBES_KERNEL_CU_
#define _MARCHING_CUBES_KERNEL_CU_

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <string.h>
#include <dpct/dpl_utils.hpp>

#include <helper_cuda.h>  // includes for helper CUDA functions
#include <helper_math.h>

#include "defines.h"
#include "tables.h"

// textures containing look-up tables
dpct::image_wrapper_base_p triTex;
dpct::image_wrapper_base_p numVertsTex;

// volume data
dpct::image_wrapper_base_p volumeTex;

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable,
                                 uint **d_numVertsTable) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  checkCudaErrors(
      DPCT_CHECK_ERROR(*d_edgeTable = sycl::malloc_device<uint>(256, q_ct1)));
  checkCudaErrors(DPCT_CHECK_ERROR(
      q_ct1.memcpy((void *)*d_edgeTable, (void *)edgeTable, 256 * sizeof(uint))
          .wait()));
  dpct::image_channel channelDesc =
      /*
      DPCT1059:0: SYCL only supports 4-channel image format. Adjust the code.
      */
      dpct::image_channel(32, 0, 0, 0,
                          dpct::image_channel_data_type::unsigned_int);

  checkCudaErrors(DPCT_CHECK_ERROR(*d_triTable = sycl::malloc_device<uint>(
                                       256 * 16, q_ct1)));
  checkCudaErrors(
      DPCT_CHECK_ERROR(q_ct1
                           .memcpy((void *)*d_triTable, (void *)triTable,
                                   256 * 16 * sizeof(uint))
                           .wait()));

  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data(*d_triTable, 256 * 16 * sizeof(uint), channelDesc);

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::clamp_to_edge,
               sycl::filtering_mode::nearest,
               sycl::coordinate_normalization_mode::unnormalized);
  /*
  DPCT1062:1: SYCL Image doesn't support normalized read mode.
  */

  checkCudaErrors(
      DPCT_CHECK_ERROR(triTex = dpct::create_image_wrapper(texRes, texDescr)));

  checkCudaErrors(DPCT_CHECK_ERROR(*d_numVertsTable =
                                       sycl::malloc_device<uint>(256, q_ct1)));
  checkCudaErrors(
      DPCT_CHECK_ERROR(q_ct1
                           .memcpy((void *)*d_numVertsTable,
                                   (void *)numVertsTable, 256 * sizeof(uint))
                           .wait()));

  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::linear);
  texRes.set_data_ptr(*d_numVertsTable);
  texRes.set_x(256 * sizeof(uint));
  texRes.set_channel(channelDesc);

  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::clamp_to_edge,
               sycl::filtering_mode::nearest,
               sycl::coordinate_normalization_mode::unnormalized);
  /*
  DPCT1062:2: SYCL Image doesn't support normalized read mode.
  */

  checkCudaErrors(DPCT_CHECK_ERROR(
      numVertsTex = dpct::create_image_wrapper(texRes, texDescr)));
}

extern "C" void createVolumeTexture(uchar *d_volume, size_t buffSize) {
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::linear);
  texRes.set_data_ptr(d_volume);
  texRes.set_x(buffSize);
  texRes.set_channel(dpct::image_channel(
      8, 0, 0, 0, dpct::image_channel_data_type::unsigned_int));

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::clamp_to_edge,
               sycl::filtering_mode::nearest,
               sycl::coordinate_normalization_mode::unnormalized);
  /*
  DPCT1062:4: SYCL Image doesn't support normalized read mode.
  */

  checkCudaErrors(DPCT_CHECK_ERROR(
      volumeTex = dpct::create_image_wrapper(texRes, texDescr)));
}

extern "C" void destroyAllTextureObjects() {
  checkCudaErrors(DPCT_CHECK_ERROR(delete triTex));
  checkCudaErrors(DPCT_CHECK_ERROR(delete numVertsTex));
  checkCudaErrors(DPCT_CHECK_ERROR(delete volumeTex));
}

// an interesting field function
float tangle(float x, float y, float z) {
  x *= 3.0f;
  y *= 3.0f;
  z *= 3.0f;
  return (x * x * x * x - 5.0f * x * x + y * y * y * y - 5.0f * y * y +
          z * z * z * z - 5.0f * z * z + 11.8f) * 0.2f + 0.5f;
}

// evaluate field function at point
float fieldFunc(sycl::float3 p) { return tangle(p.x(), p.y(), p.z()); }

// evaluate field function at a point
// returns value and gradient in float4
sycl::float4 fieldFunc4(sycl::float3 p) {
  float v = tangle(p.x(), p.y(), p.z());
  const float d = 0.001f;
  float dx = tangle(p.x() + d, p.y(), p.z()) - v;
  float dy = tangle(p.x(), p.y() + d, p.z()) - v;
  float dz = tangle(p.x(), p.y(), p.z() + d) - v;
  return sycl::float4(dx, dy, dz, v);
}

// sample volume data set at a point
/*
DPCT1050:15: The template argument of the image_accessor_ext could not be
deduced. You need to update this code.
*/
float sampleVolume(
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        volumeTex,
    uchar *data, sycl::uint3 p, sycl::uint3 gridSize) {
  p.x() = sycl::min(p.x(), gridSize.x() - 1);
  p.y() = sycl::min(p.y(), gridSize.y() - 1);
  p.z() = sycl::min(p.z(), gridSize.z() - 1);
  uint i =
      (p.z() * gridSize.x() * gridSize.y()) + (p.y() * gridSize.x()) + p.x();
  //    return (float) data[i] / 255.0f;
  return tex1Dfetch<float>(volumeTex, i);
}

// compute position in 3d grid from 1d index
// only works for power of 2 sizes
sycl::uint3 calcGridPos(uint i, sycl::uint3 gridSizeShift,
                        sycl::uint3 gridSizeMask) {
  sycl::uint3 gridPos;
  gridPos.x() = i & gridSizeMask.x();
  gridPos.y() = (i >> gridSizeShift.y()) & gridSizeMask.y();
  gridPos.z() = (i >> gridSizeShift.z()) & gridSizeMask.z();
  return gridPos;
}

// classify voxel based on number of vertices it will generate
// one thread per voxel
void classifyVoxel(
    uint *voxelVerts, uint *voxelOccupied, uchar *volume, sycl::uint3 gridSize,
    sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask, uint numVoxels,
    sycl::float3 voxelSize, float isoValue,
    /*
    DPCT1050:16: The template argument of the image_accessor_ext could not be
    deduced. You need to update this code.
    */
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        numVertsTex,
    /*
    DPCT1050:17: The template argument of the image_accessor_ext could not be
    deduced. You need to update this code.
    */
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        volumeTex,
    const sycl::nd_item<3> &item_ct1) {
  uint blockId = sycl::mul24((int)item_ct1.get_group(1),
                             (int)item_ct1.get_group_range(2)) +
                 item_ct1.get_group(2);
  uint i = sycl::mul24((int)blockId, (int)item_ct1.get_local_range(2)) +
           item_ct1.get_local_id(2);

  sycl::uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

// read field values at neighbouring grid vertices
#if SAMPLE_VOLUME
  float field[8];
  field[0] = sampleVolume(volumeTex, volume, gridPos, gridSize);
  field[1] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 0), gridSize);
  field[2] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 0), gridSize);
  field[3] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 0), gridSize);
  field[4] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 0, 1), gridSize);
  field[5] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 1), gridSize);
  field[6] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 1), gridSize);
  field[7] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 1), gridSize);
#else
  float3 p;
  p.x = -1.0f + (gridPos.x * voxelSize.x);
  p.y = -1.0f + (gridPos.y * voxelSize.y);
  p.z = -1.0f + (gridPos.z * voxelSize.z);

  float field[8];
  field[0] = fieldFunc(p);
  field[1] = fieldFunc(p + make_float3(voxelSize.x, 0, 0));
  field[2] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, 0));
  field[3] = fieldFunc(p + make_float3(0, voxelSize.y, 0));
  field[4] = fieldFunc(p + make_float3(0, 0, voxelSize.z));
  field[5] = fieldFunc(p + make_float3(voxelSize.x, 0, voxelSize.z));
  field[6] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z));
  field[7] = fieldFunc(p + make_float3(0, voxelSize.y, voxelSize.z));
#endif

  // calculate flag indicating if each vertex is inside or outside isosurface
  uint cubeindex;
  cubeindex = uint(field[0] < isoValue);
  cubeindex += uint(field[1] < isoValue) * 2;
  cubeindex += uint(field[2] < isoValue) * 4;
  cubeindex += uint(field[3] < isoValue) * 8;
  cubeindex += uint(field[4] < isoValue) * 16;
  cubeindex += uint(field[5] < isoValue) * 32;
  cubeindex += uint(field[6] < isoValue) * 64;
  cubeindex += uint(field[7] < isoValue) * 128;

  // read number of vertices from texture
  uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

  if (i < numVoxels) {
    voxelVerts[i] = numVerts;
    voxelOccupied[i] = (numVerts > 0);
  }
}

extern "C" void launch_classifyVoxel(sycl::range<3> grid,
                                     sycl::range<3> threads, uint *voxelVerts,
                                     uint *voxelOccupied, uchar *volume,
                                     sycl::uint3 gridSize,
                                     sycl::uint3 gridSizeShift,
                                     sycl::uint3 gridSizeMask, uint numVoxels,
                                     sycl::float3 voxelSize, float isoValue) {
  // calculate number of vertices need per voxel
  /*
  DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  /*
  DPCT1050:9: The template argument of the image_accessor_ext could not be
  deduced. You need to update this code.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    auto numVertsTex_acc = static_cast<dpct::image_wrapper<
        dpct_placeholder /*Fix the type manually*/, 1> *>(numVertsTex)
                               ->get_access(cgh);
    auto volumeTex_acc = static_cast<dpct::image_wrapper<
        dpct_placeholder /*Fix the type manually*/, 1> *>(volumeTex)
                             ->get_access(cgh);

    auto numVertsTex_smpl = numVertsTex->get_sampler();
    auto volumeTex_smpl = volumeTex->get_sampler();

    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       classifyVoxel(
                           voxelVerts, voxelOccupied, volume, gridSize,
                           gridSizeShift, gridSizeMask, numVoxels, voxelSize,
                           isoValue,
                           dpct::image_accessor_ext<
                               dpct_placeholder /*Fix the type manually*/, 1>(
                               numVertsTex_smpl, numVertsTex_acc),
                           dpct::image_accessor_ext<
                               dpct_placeholder /*Fix the type manually*/, 1>(
                               volumeTex_smpl, volumeTex_acc),
                           item_ct1);
                     });
  });
  getLastCudaError("classifyVoxel failed");
}

// compact voxel array
void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied,
                              uint *voxelOccupiedScan, uint numVoxels,
                              const sycl::nd_item<3> &item_ct1) {
  uint blockId = sycl::mul24((int)item_ct1.get_group(1),
                             (int)item_ct1.get_group_range(2)) +
                 item_ct1.get_group(2);
  uint i = sycl::mul24((int)blockId, (int)item_ct1.get_local_range(2)) +
           item_ct1.get_local_id(2);

  if (voxelOccupied[i] && (i < numVoxels)) {
    compactedVoxelArray[voxelOccupiedScan[i]] = i;
  }
}

extern "C" void launch_compactVoxels(sycl::range<3> grid,
                                     sycl::range<3> threads,
                                     uint *compactedVoxelArray,
                                     uint *voxelOccupied,
                                     uint *voxelOccupiedScan, uint numVoxels) {
  /*
  DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(grid * threads, threads),
      [=](sycl::nd_item<3> item_ct1) {
        compactVoxels(compactedVoxelArray, voxelOccupied, voxelOccupiedScan,
                      numVoxels, item_ct1);
      });
  getLastCudaError("compactVoxels failed");
}

// compute interpolated vertex along an edge
sycl::float3 vertexInterp(float isolevel, sycl::float3 p0, sycl::float3 p1,
                          float f0, float f1) {
  float t = (isolevel - f0) / (f1 - f0);
  return lerp(p0, p1, t);
}

// compute interpolated vertex position and normal along an edge
void vertexInterp2(float isolevel, sycl::float3 p0, sycl::float3 p1,
                   sycl::float4 f0, sycl::float4 f1, sycl::float3 &p,
                   sycl::float3 &n) {
  float t = (isolevel - f0.w()) / (f1.w() - f0.w());
  p = lerp(p0, p1, t);
  n.x() = lerp(f0.x(), f1.x(), t);
  n.y() = lerp(f0.y(), f1.y(), t);
  n.z() = lerp(f0.z(), f1.z(), t);
  //    n = normalize(n);
}

// generate triangles for each voxel using marching cubes
// interpolates normals from field function
/*
DPCT1110:7: The total declared local variable size in device function
generateTriangles exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void generateTriangles(
    sycl::float4 *pos, sycl::float4 *norm, uint *compactedVoxelArray,
    uint *numVertsScanned, sycl::uint3 gridSize, sycl::uint3 gridSizeShift,
    sycl::uint3 gridSizeMask, sycl::float3 voxelSize, float isoValue,
    uint activeVoxels, uint maxVerts,
    /*
    DPCT1050:18: The template argument of the image_accessor_ext could not be
    deduced. You need to update this code.
    */
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        triTex,
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        numVertsTex,
    const sycl::nd_item<3> &item_ct1, sycl::float3 *vertlist,
    sycl::float3 *normlist) {
  uint blockId = sycl::mul24((int)item_ct1.get_group(1),
                             (int)item_ct1.get_group_range(2)) +
                 item_ct1.get_group(2);
  uint i = sycl::mul24((int)blockId, (int)item_ct1.get_local_range(2)) +
           item_ct1.get_local_id(2);

  if (i > activeVoxels - 1) {
    // can't return here because of syncthreads()
    i = activeVoxels - 1;
  }

#if SKIP_EMPTY_VOXELS
  uint voxel = compactedVoxelArray[i];
#else
  uint voxel = i;
#endif

  // compute position in 3d grid
  sycl::uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

  sycl::float3 p;
  p.x() = -1.0f + (gridPos.x() * voxelSize.x());
  p.y() = -1.0f + (gridPos.y() * voxelSize.y());
  p.z() = -1.0f + (gridPos.z() * voxelSize.z());

  // calculate cell vertex positions
  sycl::float3 v[8];
  v[0] = p;
  v[1] = p + sycl::float3(voxelSize.x(), 0, 0);
  v[2] = p + sycl::float3(voxelSize.x(), voxelSize.y(), 0);
  v[3] = p + sycl::float3(0, voxelSize.y(), 0);
  v[4] = p + sycl::float3(0, 0, voxelSize.z());
  v[5] = p + sycl::float3(voxelSize.x(), 0, voxelSize.z());
  v[6] = p + sycl::float3(voxelSize.x(), voxelSize.y(), voxelSize.z());
  v[7] = p + sycl::float3(0, voxelSize.y(), voxelSize.z());

  // evaluate field values
  sycl::float4 field[8];
  field[0] = fieldFunc4(v[0]);
  field[1] = fieldFunc4(v[1]);
  field[2] = fieldFunc4(v[2]);
  field[3] = fieldFunc4(v[3]);
  field[4] = fieldFunc4(v[4]);
  field[5] = fieldFunc4(v[5]);
  field[6] = fieldFunc4(v[6]);
  field[7] = fieldFunc4(v[7]);

  // recalculate flag
  // (this is faster than storing it in global memory)
  uint cubeindex;
  cubeindex = uint(field[0].w() < isoValue);
  cubeindex += uint(field[1].w() < isoValue) * 2;
  cubeindex += uint(field[2].w() < isoValue) * 4;
  cubeindex += uint(field[3].w() < isoValue) * 8;
  cubeindex += uint(field[4].w() < isoValue) * 16;
  cubeindex += uint(field[5].w() < isoValue) * 32;
  cubeindex += uint(field[6].w() < isoValue) * 64;
  cubeindex += uint(field[7].w() < isoValue) * 128;

// find the vertices where the surface intersects the cube

#if USE_SHARED
  // use partioned shared memory to avoid using local memory

  vertexInterp2(isoValue, v[0], v[1], field[0], field[1],
                vertlist[item_ct1.get_local_id(2)],
                normlist[item_ct1.get_local_id(2)]);
  vertexInterp2(isoValue, v[1], v[2], field[1], field[2],
                vertlist[item_ct1.get_local_id(2) + NTHREADS],
                normlist[item_ct1.get_local_id(2) + NTHREADS]);
  vertexInterp2(isoValue, v[2], v[3], field[2], field[3],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 2)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 2)]);
  vertexInterp2(isoValue, v[3], v[0], field[3], field[0],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 3)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 3)]);
  vertexInterp2(isoValue, v[4], v[5], field[4], field[5],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 4)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 4)]);
  vertexInterp2(isoValue, v[5], v[6], field[5], field[6],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 5)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 5)]);
  vertexInterp2(isoValue, v[6], v[7], field[6], field[7],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 6)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 6)]);
  vertexInterp2(isoValue, v[7], v[4], field[7], field[4],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 7)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 7)]);
  vertexInterp2(isoValue, v[0], v[4], field[0], field[4],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 8)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 8)]);
  vertexInterp2(isoValue, v[1], v[5], field[1], field[5],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 9)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 9)]);
  vertexInterp2(isoValue, v[2], v[6], field[2], field[6],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 10)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 10)]);
  vertexInterp2(isoValue, v[3], v[7], field[3], field[7],
                vertlist[item_ct1.get_local_id(2) + (NTHREADS * 11)],
                normlist[item_ct1.get_local_id(2) + (NTHREADS * 11)]);
  item_ct1.barrier(sycl::access::fence_space::local_space);

#else
  float3 vertlist[12];
  float3 normlist[12];

  vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertlist[0],
                normlist[0]);
  vertexInterp2(isoValue, v[1], v[2], field[1], field[2], vertlist[1],
                normlist[1]);
  vertexInterp2(isoValue, v[2], v[3], field[2], field[3], vertlist[2],
                normlist[2]);
  vertexInterp2(isoValue, v[3], v[0], field[3], field[0], vertlist[3],
                normlist[3]);

  vertexInterp2(isoValue, v[4], v[5], field[4], field[5], vertlist[4],
                normlist[4]);
  vertexInterp2(isoValue, v[5], v[6], field[5], field[6], vertlist[5],
                normlist[5]);
  vertexInterp2(isoValue, v[6], v[7], field[6], field[7], vertlist[6],
                normlist[6]);
  vertexInterp2(isoValue, v[7], v[4], field[7], field[4], vertlist[7],
                normlist[7]);

  vertexInterp2(isoValue, v[0], v[4], field[0], field[4], vertlist[8],
                normlist[8]);
  vertexInterp2(isoValue, v[1], v[5], field[1], field[5], vertlist[9],
                normlist[9]);
  vertexInterp2(isoValue, v[2], v[6], field[2], field[6], vertlist[10],
                normlist[10]);
  vertexInterp2(isoValue, v[3], v[7], field[3], field[7], vertlist[11],
                normlist[11]);
#endif

  // output triangle vertices
  uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

  for (int i = 0; i < numVerts; i++) {
    uint edge = tex1Dfetch<uint>(triTex, cubeindex * 16 + i);

    uint index = numVertsScanned[voxel] + i;

    if (index < maxVerts) {
#if USE_SHARED
      pos[index] = make_float4(
          vertlist[(edge * NTHREADS) + item_ct1.get_local_id(2)], 1.0f);
      norm[index] = make_float4(
          normlist[(edge * NTHREADS) + item_ct1.get_local_id(2)], 0.0f);
#else
      pos[index] = make_float4(vertlist[edge], 1.0f);
      norm[index] = make_float4(normlist[edge], 0.0f);
#endif
    }
  }
}

extern "C" void launch_generateTriangles(
    sycl::range<3> grid, sycl::range<3> threads, sycl::float4 *pos,
    sycl::float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
    sycl::uint3 gridSize, sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask,
    sycl::float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts) {
  /*
  DPCT1050:12: The template argument of the image_accessor_ext could not be
  deduced. You need to update this code.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:10: '12 * NTHREADS' expression was replaced with a value. Modify
    the code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<sycl::float3, 1> vertlist_acc_ct1(
        sycl::range<1>(384 /*12 * NTHREADS*/), cgh);
    /*
    DPCT1101:11: '12 * NTHREADS' expression was replaced with a value. Modify
    the code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<sycl::float3, 1> normlist_acc_ct1(
        sycl::range<1>(384 /*12 * NTHREADS*/), cgh);

    auto triTex_acc = static_cast<dpct::image_wrapper<
        dpct_placeholder /*Fix the type manually*/, 1> *>(triTex)
                          ->get_access(cgh);
    auto numVertsTex_acc = static_cast<dpct::image_wrapper<
        dpct_placeholder /*Fix the type manually*/, 1> *>(numVertsTex)
                               ->get_access(cgh);

    auto triTex_smpl = triTex->get_sampler();
    auto numVertsTex_smpl = numVertsTex->get_sampler();

    cgh.parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, NTHREADS),
                          sycl::range<3>(1, 1, NTHREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          generateTriangles(
              pos, norm, compactedVoxelArray, numVertsScanned, gridSize,
              gridSizeShift, gridSizeMask, voxelSize, isoValue, activeVoxels,
              maxVerts,
              dpct::image_accessor_ext<
                  dpct_placeholder /*Fix the type manually*/, 1>(triTex_smpl,
                                                                 triTex_acc),
              dpct::image_accessor_ext<
                  dpct_placeholder /*Fix the type manually*/, 1>(
                  numVertsTex_smpl, numVertsTex_acc),
              item_ct1,
              vertlist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              normlist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
  getLastCudaError("generateTriangles failed");
}

// calculate triangle normal
sycl::float3 calcNormal(sycl::float3 *v0, sycl::float3 *v1, sycl::float3 *v2) {
  sycl::float3 edge0 = *v1 - *v0;
  sycl::float3 edge1 = *v2 - *v0;
  // note - it's faster to perform normalization in vertex shader rather than
  // here
  return cross(edge0, edge1);
}

// version that calculates flat surface normal for each triangle
/*
DPCT1110:8: The total declared local variable size in device function
generateTriangles2 exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void generateTriangles2(
    sycl::float4 *pos, sycl::float4 *norm, uint *compactedVoxelArray,
    uint *numVertsScanned, uchar *volume, sycl::uint3 gridSize,
    sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask, sycl::float3 voxelSize,
    float isoValue, uint activeVoxels, uint maxVerts,
    /*
    DPCT1050:19: The template argument of the image_accessor_ext could not be
    deduced. You need to update this code.
    */
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        triTex,
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        numVertsTex,
    /*
    DPCT1050:20: The template argument of the image_accessor_ext could not be
    deduced. You need to update this code.
    */
    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
        volumeTex,
    const sycl::nd_item<3> &item_ct1, sycl::float3 *vertlist) {
  uint blockId = sycl::mul24((int)item_ct1.get_group(1),
                             (int)item_ct1.get_group_range(2)) +
                 item_ct1.get_group(2);
  uint i = sycl::mul24((int)blockId, (int)item_ct1.get_local_range(2)) +
           item_ct1.get_local_id(2);

  if (i > activeVoxels - 1) {
    i = activeVoxels - 1;
  }

#if SKIP_EMPTY_VOXELS
  uint voxel = compactedVoxelArray[i];
#else
  uint voxel = i;
#endif

  // compute position in 3d grid
  sycl::uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

  sycl::float3 p;
  p.x() = -1.0f + (gridPos.x() * voxelSize.x());
  p.y() = -1.0f + (gridPos.y() * voxelSize.y());
  p.z() = -1.0f + (gridPos.z() * voxelSize.z());

  // calculate cell vertex positions
  sycl::float3 v[8];
  v[0] = p;
  v[1] = p + sycl::float3(voxelSize.x(), 0, 0);
  v[2] = p + sycl::float3(voxelSize.x(), voxelSize.y(), 0);
  v[3] = p + sycl::float3(0, voxelSize.y(), 0);
  v[4] = p + sycl::float3(0, 0, voxelSize.z());
  v[5] = p + sycl::float3(voxelSize.x(), 0, voxelSize.z());
  v[6] = p + sycl::float3(voxelSize.x(), voxelSize.y(), voxelSize.z());
  v[7] = p + sycl::float3(0, voxelSize.y(), voxelSize.z());

#if SAMPLE_VOLUME
  float field[8];
  field[0] = sampleVolume(volumeTex, volume, gridPos, gridSize);
  field[1] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 0), gridSize);
  field[2] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 0), gridSize);
  field[3] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 0), gridSize);
  field[4] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 0, 1), gridSize);
  field[5] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 0, 1), gridSize);
  field[6] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(1, 1, 1), gridSize);
  field[7] =
      sampleVolume(volumeTex, volume, gridPos + sycl::uint3(0, 1, 1), gridSize);
#else
  // evaluate field values
  float field[8];
  field[0] = fieldFunc(v[0]);
  field[1] = fieldFunc(v[1]);
  field[2] = fieldFunc(v[2]);
  field[3] = fieldFunc(v[3]);
  field[4] = fieldFunc(v[4]);
  field[5] = fieldFunc(v[5]);
  field[6] = fieldFunc(v[6]);
  field[7] = fieldFunc(v[7]);
#endif

  // recalculate flag
  uint cubeindex;
  cubeindex = uint(field[0] < isoValue);
  cubeindex += uint(field[1] < isoValue) * 2;
  cubeindex += uint(field[2] < isoValue) * 4;
  cubeindex += uint(field[3] < isoValue) * 8;
  cubeindex += uint(field[4] < isoValue) * 16;
  cubeindex += uint(field[5] < isoValue) * 32;
  cubeindex += uint(field[6] < isoValue) * 64;
  cubeindex += uint(field[7] < isoValue) * 128;

// find the vertices where the surface intersects the cube

#if USE_SHARED
  // use shared memory to avoid using local

  vertlist[item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
  vertlist[NTHREADS + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
  vertlist[(NTHREADS * 2) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
  vertlist[(NTHREADS * 3) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
  vertlist[(NTHREADS * 4) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
  vertlist[(NTHREADS * 5) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
  vertlist[(NTHREADS * 6) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
  vertlist[(NTHREADS * 7) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
  vertlist[(NTHREADS * 8) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
  vertlist[(NTHREADS * 9) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
  vertlist[(NTHREADS * 10) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
  vertlist[(NTHREADS * 11) + item_ct1.get_local_id(2)] =
      vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
  item_ct1.barrier(sycl::access::fence_space::local_space);
#else

  float3 vertlist[12];

  vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
  vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
  vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
  vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

  vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
  vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
  vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
  vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

  vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
  vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
  vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
  vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
#endif

  // output triangle vertices
  uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

  for (int i = 0; i < numVerts; i += 3) {
    uint index = numVertsScanned[voxel] + i;

    sycl::float3 *v[3];
    uint edge;
    edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + i);
#if USE_SHARED
    v[0] = &vertlist[(edge * NTHREADS) + item_ct1.get_local_id(2)];
#else
    v[0] = &vertlist[edge];
#endif

    edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + i + 1);
#if USE_SHARED
    v[1] = &vertlist[(edge * NTHREADS) + item_ct1.get_local_id(2)];
#else
    v[1] = &vertlist[edge];
#endif

    edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + i + 2);
#if USE_SHARED
    v[2] = &vertlist[(edge * NTHREADS) + item_ct1.get_local_id(2)];
#else
    v[2] = &vertlist[edge];
#endif

    // calculate triangle surface normal
    sycl::float3 n = calcNormal(v[0], v[1], v[2]);

    if (index < (maxVerts - 3)) {
      pos[index] = make_float4(*v[0], 1.0f);
      norm[index] = make_float4(n, 0.0f);

      pos[index + 1] = make_float4(*v[1], 1.0f);
      norm[index + 1] = make_float4(n, 0.0f);

      pos[index + 2] = make_float4(*v[2], 1.0f);
      norm[index + 2] = make_float4(n, 0.0f);
    }
  }
}

extern "C" void launch_generateTriangles2(
    sycl::range<3> grid, sycl::range<3> threads, sycl::float4 *pos,
    sycl::float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
    uchar *volume, sycl::uint3 gridSize, sycl::uint3 gridSizeShift,
    sycl::uint3 gridSizeMask, sycl::float3 voxelSize, float isoValue,
    uint activeVoxels, uint maxVerts) {
  /*
  DPCT1050:14: The template argument of the image_accessor_ext could not be
  deduced. You need to update this code.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:13: '12 * NTHREADS' expression was replaced with a value. Modify
    the code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<sycl::float3, 1> vertlist_acc_ct1(
        sycl::range<1>(384 /*12 * NTHREADS*/), cgh);

    auto triTex_acc = static_cast<dpct::image_wrapper<
        dpct_placeholder /*Fix the type manually*/, 1> *>(triTex)
                          ->get_access(cgh);
    auto numVertsTex_acc = static_cast<dpct::image_wrapper<
        dpct_placeholder /*Fix the type manually*/, 1> *>(numVertsTex)
                               ->get_access(cgh);
    auto volumeTex_acc = static_cast<dpct::image_wrapper<
        dpct_placeholder /*Fix the type manually*/, 1> *>(volumeTex)
                             ->get_access(cgh);

    auto triTex_smpl = triTex->get_sampler();
    auto numVertsTex_smpl = numVertsTex->get_sampler();
    auto volumeTex_smpl = volumeTex->get_sampler();

    cgh.parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, NTHREADS),
                          sycl::range<3>(1, 1, NTHREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          generateTriangles2(
              pos, norm, compactedVoxelArray, numVertsScanned, volume, gridSize,
              gridSizeShift, gridSizeMask, voxelSize, isoValue, activeVoxels,
              maxVerts,
              dpct::image_accessor_ext<
                  dpct_placeholder /*Fix the type manually*/, 1>(triTex_smpl,
                                                                 triTex_acc),
              dpct::image_accessor_ext<
                  dpct_placeholder /*Fix the type manually*/, 1>(
                  numVertsTex_smpl, numVertsTex_acc),
              dpct::image_accessor_ext<
                  dpct_placeholder /*Fix the type manually*/, 1>(volumeTex_smpl,
                                                                 volumeTex_acc),
              item_ct1,
              vertlist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
  getLastCudaError("generateTriangles2 failed");
}

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
                                  unsigned int numElements) {
  thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
                         thrust::device_ptr<unsigned int>(input + numElements),
                         thrust::device_ptr<unsigned int>(output));
}

#endif
