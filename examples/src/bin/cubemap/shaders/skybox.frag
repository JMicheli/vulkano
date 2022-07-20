// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
#version 450

layout(location = 0) in vec3 v_pos;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 1.0, 1.0);
}
