// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 v_pos;

void main() {
    v_pos = position;
    gl_Position = vec4(v_pos, 1.0);
}
