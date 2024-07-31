#ifndef _MARCHING_CUBES_KERNEL_SYCL_
#define _MARCHING_CUBES_KERNEL_SYCL_

#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <cstring>
#include "defines.h"
#include "tables.h"

// textures containing look-up tables
sycl::buffer<uint, 1> *triTableBuf;
sycl::buffer<uint, 1> *numVertsTableBuf;

// volume data
sycl::buffer<uchar, 1> *volumeBuf;

extern "C" void allocateTextures(sycl::queue &q, uint **d_edgeTable, uint **d_triTable,
                                 uint **d_numVertsTable) {
  *d_edgeTable = static_cast<uint *>(sycl::malloc_device(256 * sizeof(uint), q));
  q.memcpy(*d_edgeTable, edgeTable, 256 * sizeof(uint)).wait();

  *d_triTable = static_cast<uint *>(sycl::malloc_device(256 * 16 * sizeof(uint), q));
  q.memcpy(*d_triTable, triTable, 256 * 16 * sizeof(uint)).wait();

  triTableBuf = new sycl::buffer<uint, 1>(*d_triTable, sycl::range<1>(256 * 16));

  *d_numVertsTable = static_cast<uint *>(sycl::malloc_device(256 * sizeof(uint), q));
  q.memcpy(*d_numVertsTable, numVertsTable, 256 * sizeof(uint)).wait();

  numVertsTableBuf = new sycl::buffer<uint, 1>(*d_numVertsTable, sycl::range<1>(256));
}

extern "C" void createVolumeTexture(uchar *d_volume, size_t buffSize) {
  volumeBuf = new sycl::buffer<uchar, 1>(d_volume, sycl::range<1>(buffSize));
}

extern "C" void destroyAllTextureObjects() {
  delete triTableBuf;
  delete numVertsTableBuf;
  delete volumeBuf;
}

float tangle(float x, float y, float z) {
  x *= 3.0f;
  y *= 3.0f;
  z *= 3.0f;
  return (x * x * x * x - 5.0f * x * x + y * y * y * y - 5.0f * y * y +
          z * z * z * z - 5.0f * z * z + 11.8f) * 0.2f + 0.5f;
}

float fieldFunc(sycl::float3 p) {
  return tangle(p.x(), p.y(), p.z());
}

sycl::float4 fieldFunc4(sycl::float3 p) {
  float v = tangle(p.x(), p.y(), p.z());
  const float d = 0.001f;
  float dx = tangle(p.x() + d, p.y(), p.z()) - v;
  float dy = tangle(p.x(), p.y() + d, p.z()) - v;
  float dz = tangle(p.x(), p.y(), p.z() + d) - v;
  return sycl::float4{dx, dy, dz, v};
}

float sampleVolume(sycl::accessor<uchar, 1, sycl::access_mode::read> volumeAcc, uchar *data, sycl::uint3 p, sycl::uint3 gridSize) {
  p.x() = sycl::min(p.x(), gridSize.x() - 1);
  p.y() = sycl::min(p.y(), gridSize.y() - 1);
  p.z() = sycl::min(p.z(), gridSize.z() - 1);
  uint i = (p.z() * gridSize.x() * gridSize.y()) + (p.y() * gridSize.x()) + p.x();
  return volumeAcc[i];
}

sycl::uint3 calcGridPos(uint i, sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask) {
  sycl::uint3 gridPos;
  gridPos.x() = i & gridSizeMask.x();
  gridPos.y() = (i >> gridSizeShift.y()) & gridSizeMask.y();
  gridPos.z() = (i >> gridSizeShift.z()) & gridSizeMask.z();
  return gridPos;
}

void classifyVoxel(uint *voxelVerts, uint *voxelOccupied, uchar *volume,
                   sycl::uint3 gridSize, sycl::uint3 gridSizeShift,
                   sycl::uint3 gridSizeMask, uint numVoxels,
                   sycl::float3 voxelSize, float isoValue,
                   sycl::accessor<uint, 1, sycl::access_mode::read> numVertsAcc,
                   sycl::accessor<uchar, 1, sycl::access_mode::read> volumeAcc,
                   //sycl::nd_item<3> item_ct1)
		   sycl::id<3> idx) {
  //uint blockId = item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2);
  //uint i = blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  uint i = idx[2] * gridSize.x() * gridSize.y() + idx[1] * gridSize.x() + idx[0];

  sycl::uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

  float field[8];
  field[0] = sampleVolume(volumeAcc, volume, gridPos, gridSize);
  field[1] = sampleVolume(volumeAcc, volume, gridPos + sycl::uint3(1, 0, 0), gridSize);
  field[2] = sampleVolume(volumeAcc, volume, gridPos + sycl::uint3(1, 1, 0), gridSize);
  field[3] = sampleVolume(volumeAcc, volume, gridPos + sycl::uint3(0, 1, 0), gridSize);
  field[4] = sampleVolume(volumeAcc, volume, gridPos + sycl::uint3(0, 0, 1), gridSize);
  field[5] = sampleVolume(volumeAcc, volume, gridPos + sycl::uint3(1, 0, 1), gridSize);
  field[6] = sampleVolume(volumeAcc, volume, gridPos + sycl::uint3(1, 1, 1), gridSize);
  field[7] = sampleVolume(volumeAcc, volume, gridPos + sycl::uint3(0, 1, 1), gridSize);

  uint cubeindex;
  cubeindex = uint(field[0] < isoValue);
  cubeindex += uint(field[1] < isoValue) * 2;
  cubeindex += uint(field[2] < isoValue) * 4;
  cubeindex += uint(field[3] < isoValue) * 8;
  cubeindex += uint(field[4] < isoValue) * 16;
  cubeindex += uint(field[5] < isoValue) * 32;
  cubeindex += uint(field[6] < isoValue) * 64;
  cubeindex += uint(field[7] < isoValue) * 128;

  uint numVerts = numVertsAcc[cubeindex];

  if (i < numVoxels) {
    voxelVerts[i] = numVerts;
    voxelOccupied[i] = (numVerts > 0);
  }
}

extern "C" void launch_classifyVoxel(sycl::queue &q, sycl::range<3> globalRange,
                                     uint *voxelVerts, uint *voxelOccupied, uchar *volume,
                                     sycl::uint3 gridSize, sycl::uint3 gridSizeShift,
                                     sycl::uint3 gridSizeMask, uint numVoxels,
                                     sycl::float3 voxelSize, float isoValue) {
  q.submit([&](sycl::handler &h) {
    auto out = sycl::stream(1024, 768, h);
    auto numVertsAcc = numVertsTableBuf -> get_access<sycl::access_mode::read>(h);
    auto volumeAcc = volumeBuf -> get_access<sycl::access_mode::read>(h);
    
    //out << "voxelVerts before kernel execution:\n[";
    //for (uint i = 0; i < numVoxels; ++i) {
    //  out << voxelVerts[i] << " ";
    //}
    //out << "]\n";

    h.parallel_for(globalRange, [=](sycl::id<3> idx) {
          classifyVoxel(voxelVerts, voxelOccupied, volume, gridSize,
                        gridSizeShift, gridSizeMask, numVoxels, voxelSize,
                        isoValue, numVertsAcc, volumeAcc, idx);
        });

    //out << "voxelVerts after kernel execution:\n[";
    //for (uint i = 0; i < numVoxels; ++i) {
    //  out << voxelVerts[i] << " ";
    //}
    //out << "]\n";
  }).wait();
}

void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied,
                   uint *voxelOccupiedScan, uint numVoxels, sycl::id<3> idx) {
  //uint blockId = item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2);
  //uint i = blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  uint i = idx[2] * numVoxels * numVoxels + idx[1] * numVoxels + idx[0];
  if (i >= numVoxels) return;
  
  if (voxelOccupied[i]) {
    compactedVoxelArray[voxelOccupiedScan[i]] = i;
  }
}

extern "C" void launch_compactVoxels(sycl::queue &q, sycl::range<3> globalRange,
                                     uint *compactedVoxelArray, uint *voxelOccupied,
                                     uint *voxelOccupiedScan, uint numVoxels) {
  q.parallel_for(globalRange, [=](sycl::id<3> idx) {
          compactVoxels(compactedVoxelArray, voxelOccupied, voxelOccupiedScan, numVoxels, idx);
        }).wait();
}

sycl::float3 vertexInterp(float isolevel, sycl::float3 p0, sycl::float3 p1, float f0, float f1) {
  float t = (isolevel - f0) / (f1 - f0);
  return p0 + t * (p1 - p0);
}

void vertexInterp2(float isolevel, sycl::float3 p0, sycl::float3 p1, sycl::float4 f0, sycl::float4 f1, sycl::float3 &p, sycl::float3 &n) {
  float t = (isolevel- f0.w()) / (f1.w() - f0.w());
  p = p0 + t * (p1 - p0);
  n.x() = f0.x() + t * (f1.x() - f0.x());
  n.y() = f0.y() + t * (f1.y() - f0.y());
  n.z() = f0.z() + t * (f1.z() - f0.z());
}

void generateTriangles(sycl::float4 *pos, sycl::float4 *norm, uint *compactedVoxelArray,
                       uint *numVertsScanned, sycl::uint3 gridSize,
                       sycl::uint3 gridSizeShift, sycl::uint3 gridSizeMask,
                       sycl::float3 voxelSize, float isoValue,
                       uint activeVoxels, uint maxVerts,
                       sycl::accessor<uint, 1, sycl::access_mode::read> triTableAcc,
                       sycl::accessor<uint, 1, sycl::access_mode::read> numVertsAcc,
                       sycl::id<3> idx) {
  //uint blockId = item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2);
  //uint i = blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  uint i = idx[2] * gridSize.x() * gridSize.y() + idx[1] * gridSize.x() + idx[0];
	
  if (i >= activeVoxels) return;

  uint voxel = compactedVoxelArray[i];

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
  cubeindex += uint(field[1].w() < isoValue) * 2;
  cubeindex += uint(field[2].w() < isoValue) * 4;
  cubeindex += uint(field[3].w() < isoValue) * 8;
  cubeindex += uint(field[4].w() < isoValue) * 16;
  cubeindex += uint(field[5].w() < isoValue) * 32;
  cubeindex += uint(field[6].w() < isoValue) * 64;
  cubeindex += uint(field[7].w() < isoValue) * 128;

  sycl::float3 vertlist[12];
  sycl::float3 normlist[12];

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

  uint numVerts = numVertsAcc[cubeindex];

  for (int i = 0; i < numVerts; i++) {
    uint edge = triTableAcc[cubeindex * 16 + i];

    uint index = numVertsScanned[voxel] + i;

    if (index < maxVerts) {
      pos[index] = sycl::float4{vertlist[edge].x(), vertlist[edge].y(), vertlist[edge].z(), 1.0f};
      norm[index] = sycl::float4{normlist[edge].x(), normlist[edge].y(), normlist[edge].z(), 0.0f};
    }
  }
}

extern "C" void launch_generateTriangles(sycl::queue &q, sycl::range<3> globalRange,
                                         sycl::float4 *pos, sycl::float4 *norm, 
					 uint *compactedVoxelArray, uint *numVertsScanned, 
					 sycl::uint3 gridSize, sycl::uint3 gridSizeShift,
                                         sycl::uint3 gridSizeMask, sycl::float3 voxelSize,
					 float isoValue, uint activeVoxels, uint maxVerts) {
  q.submit([&](sycl::handler &h) {
    auto triTableAcc = triTableBuf->get_access<sycl::access_mode::read>(h);
    auto numVertsAcc = numVertsTableBuf->get_access<sycl::access_mode::read>(h);

    h.parallel_for(globalRange, [=](sycl::id<3> idx) {
          generateTriangles(pos, norm, compactedVoxelArray, numVertsScanned,
                            gridSize, gridSizeShift, gridSizeMask, voxelSize,
                            isoValue, activeVoxels, maxVerts, triTableAcc,
                            numVertsAcc, idx);
        });
  }).wait();
}

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
                                  unsigned int numElements) {
	oneapi::dpl::exclusive_scan(oneapi::dpl::execution::dpcpp_default, input, input + numElements, output, 0);
}

#endif

