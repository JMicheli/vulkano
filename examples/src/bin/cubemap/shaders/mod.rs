pub mod mesh_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/cubemap/shaders/mesh.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod mesh_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/cubemap/shaders/mesh.frag",
    }
}

pub use mesh_vertex_shader::ty::CameraData;

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
