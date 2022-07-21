use std::sync::Arc;

use examples::Vertex;
use vulkano::{
    device::Queue,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState, vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline,
    },
    render_pass::{RenderPass, Subpass},
};

use crate::shaders::{mesh_fragment_shader, mesh_vertex_shader};

pub struct RenderPipeline {
    pub pipeline: Arc<GraphicsPipeline>,
}

impl RenderPipeline {
    pub fn new(gfx_queue: &Arc<Queue>, render_pass: &Arc<RenderPass>) -> Self {
        let pipeline = GraphicsPipeline::start()
            // We set the subpass to use for this pipeline
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // We set the input state - what an input vertex will look like to the pipeline
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            // We link a vertex shader which will run for each vertex
            .vertex_shader(
                // The vertex shader module contains a load function to upload it to the GPU
                mesh_vertex_shader::load(gfx_queue.device().clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            // // We set the rasterizer state for backface culling.
            // .rasterization_state(
            //     RasterizationState::new()
            //         // The rasterizer will treat clockwise-wound triangles as front-facing...
            //         .front_face(FrontFace::CounterClockwise)
            //         // ...and cull (not execute the fragment shader for) back-facing triangles
            //         .cull_mode(CullMode::Back),
            // )
            // We'll use a simple depth test to ensure correct order of fragments
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            // Use a resizable viewport set to draw over the entire window
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            // We link the fragment shader which will run for each fragment (output pixel)
            .fragment_shader(
                mesh_fragment_shader::load(gfx_queue.device().clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap(),
                (),
            )
            // We call build() to create the pipeline object and return a handle
            .build(gfx_queue.device().clone())
            .unwrap();

        Self { pipeline }
    }
}
