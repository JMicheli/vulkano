pub mod mesh_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/cubemap/shaders/mesh.vert",
    }
}

pub mod mesh_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/cubemap/shaders/mesh.frag",
    }
}

pub mod skybox_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/cubemap/shaders/skybox.vert",
    }
}

pub mod skybox_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/cubemap/shaders/skybox.frag",
    }
}
