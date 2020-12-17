/* Copyright 2020 The TensorFlow Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_KERNELS_RASTERIZE_TRIANGLES_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_KERNELS_RASTERIZE_TRIANGLES_H_

#include <memory>
#include <vector>

#include "absl/base/integral_types.h"

// Simple wrapper around the core rasterizer that does not use tensorflow.
// See rasterize_triangles_impl.h for a more detailed description of these
// arguments.
//
// vertices: A flattened 2D array with 4*vertex_count elements.
// triangles: A flattened 2D array with 3*triangle_count elements.
// image_width: The width of the output image.
// image_height: The height of the output image.
// triangle_ids: A flattened 2D array with image_height*image_width elements
//     in row-major order. At return, contains the triangle ids. Will be
//     initialized and cleared to 0 if size does not match expected image size.
//     If the size matches the buffer is not cleared.
// z_buffer: A flattened 2D array with image_height*image_width elements in row-
//     major order. At return, contains the normalized device Z coordinates of
//     the rendered triangles. Allocated and cleared to "far" (1.0) if size does
//     not match image size; otherwise not cleared.
// barycentric_coordinates: A flattened 3D array with image_height*image_width*3
//     elements in row-major order. At return, contains the triplet of
//     barycentric coordinates. Will be initialized and cleared to 0 if size
//     does not match expected image size, otherwise not cleared.
void RasterizeTriangles(const std::vector<float>& vertices,
                        const std::vector<int32>& triangles, int32 image_width,
                        int32 image_height, std::vector<int32>* triangle_ids,
                        std::vector<float>* z_buffer,
                        std::vector<float>* barycentric_coordinates);

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_KERNELS_RASTERIZE_TRIANGLES_H_
