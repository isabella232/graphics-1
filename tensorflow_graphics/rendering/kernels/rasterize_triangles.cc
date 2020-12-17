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
#include "tensorflow_graphics/rendering/kernels/rasterize_triangles.h"

#include "base/logging.h"
#include "tensorflow_graphics/rendering/kernels/rasterize_triangles_impl.h"

namespace {

constexpr float kClearDepth = 1.0;

}

void RasterizeTriangles(const std::vector<float>& vertices,
                        const std::vector<int32>& triangles, int32 image_width,
                        int32 image_height, std::vector<int32>* triangle_ids,
                        std::vector<float>* z_buffer,
                        std::vector<float>* barycentric_coordinates) {
  CHECK_EQ(vertices.size() % 4, 0)
      << "vertices must have size divisible by 4. Got: " << vertices.size();
  CHECK_EQ(triangles.size() % 3, 0)
      << "triangles must have size divisible by 3. Got: " << triangles.size();
  CHECK_NE(z_buffer, nullptr) << "z_buffer cannot be nullptr.";
  CHECK_NE(barycentric_coordinates, nullptr)
      << "barycentric_coordinates cannot be nullptr.";
  const int triangle_count = triangles.size() / 3;
  const int num_pixels = image_width * image_height;
  triangle_ids->resize(num_pixels, 0);
  z_buffer->resize(num_pixels, kClearDepth);
  barycentric_coordinates->resize(num_pixels * 3, 0);

  // TODO(dvlasic): pass in number of layers as arg.
  const int32 num_layers = 1;
  RasterizeTrianglesImpl(vertices.data(), triangles.data(), triangle_count,
                         image_width, image_height, num_layers,
                         FaceCullingMode::kNone, triangle_ids->data(),
                         z_buffer->data(), barycentric_coordinates->data());
}
