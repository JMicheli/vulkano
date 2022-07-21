// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::render_pipeline::RenderPipeline;
use crate::shaders::mesh_vertex_shader::ty::CameraData;
use cgmath::{Angle, Deg, Matrix4, Point3, Rad, Vector3};
use examples::Vertex;
use image::io::Reader as ImageReader;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
    PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Queue;
use vulkano::format::ClearValue;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageDimensions, ImmutableImage};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::sync::{GpuFuture, NowFuture};
use vulkano_util::renderer::SwapchainImageView;

// We'll yoink this test model data from the examples lib module (it's the Utah Teapot)
use examples::{INDICES as TEST_MODEL_INDICES, VERTICES as TEST_MODEL_VERTICES};

// These are the paths for the skybox textures
const SKYBOX_TOP: &str = "examples/src/bin/cubemap/skybox/top.jpg";
const SKYBOX_BOTTOM: &str = "examples/src/bin/cubemap/skybox/bottom.jpg";
const SKYBOX_LEFT: &str = "examples/src/bin/cubemap/skybox/left.jpg";
const SKYBOX_RIGHT: &str = "examples/src/bin/cubemap/skybox/right.jpg";
const SKYBOX_FRONT: &str = "examples/src/bin/cubemap/skybox/front.jpg";
const SKYBOX_BACK: &str = "examples/src/bin/cubemap/skybox/back.jpg";

// These values will be used in the camera
const CAMERA_ROTATION_SPEED: f32 = 60.0;
const NEAR_DISTANCE: f32 = 0.01;
const FIELD_OF_VIEW_ANGLE: f32 = 90.0;
const FAR_DISTANCE: f32 = 500.0;
const UP_DIRECTION: Vector3<f32> = Vector3 {
    x: 0.0,
    y: -1.0,
    z: 0.0,
};

// The clear value used for the swapchain image. In a skyboxed scene,
// this should never be visible, so I've chosen an obnoxious magenta.
const BACKGROUND_COLOR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// App for displaying a skyboxed scene
pub struct CubeMapApp {
    // Vulkano members
    gfx_queue: Arc<Queue>,
    render_pipeline: RenderPipeline,
    render_pass: Arc<RenderPass>,
    framebuffer: Option<Arc<Framebuffer>>,

    // Render asset data
    test_model: Model,
    camera: Camera,
    skybox: Skybox,
    skybox_upload_future: Option<Box<dyn GpuFuture>>,

    /// Time tracking, useful for frame independent movement
    time: Instant,
    dt: f32,
    dt_sum: f32,
    frame_count: f32,
    avg_fps: f32,
}

impl CubeMapApp {
    pub fn new(gfx_queue: Arc<Queue>, output_format: vulkano::format::Format) -> Self {
        // Set up vulkano objects for rendering
        // A render pass describes the data that will be written during a single
        // render command (a pass).
        let render_pass = vulkano::single_pass_renderpass!(gfx_queue.device().clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: output_format,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: vulkano::format::Format::D16_UNORM,
                    samples: 1,
                  }
            },
            pass: {
                    color: [color],
                    depth_stencil: {depth}
            }
        )
        .unwrap();

        // Pipeline
        let render_pipeline = RenderPipeline::new(&gfx_queue, &render_pass);

        // Set up scene resources
        let test_model = Model::teapot(&gfx_queue);
        let camera = Camera::default();
        let (skybox, skybox_upload_future) = Skybox::new(&gfx_queue);

        Self {
            gfx_queue: gfx_queue.clone(),
            render_pipeline,
            render_pass,
            framebuffer: None,

            test_model,
            camera,
            skybox,
            skybox_upload_future: Some(skybox_upload_future.boxed()),

            time: Instant::now(),
            dt: 0.0,
            dt_sum: 0.0,
            frame_count: 0.0,
            avg_fps: 0.0,
        }
    }

    pub fn render(&mut self, before_future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        // Retrieve framebuffer to draw to. We set this in the main loop,
        // so the panic should never happen.
        let framebuffer = match self.framebuffer.take() {
            Some(fb) => fb,
            None => panic!("No framebuffer set in main loop before render call"),
        };

        // Create a command buffer builder to generate the render commands
        let mut cb_builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Start the renderpass - this call is needed to start rendering. Here we use
        // a Framebuffer (and thus RenderPass data), but with the dynamic rendering
        // extension, this wouldn't be needed. See triangle-v1_3 for more information.
        let mut render_pass_begin_info = RenderPassBeginInfo::framebuffer(framebuffer.clone());
        render_pass_begin_info.clear_values = vec![
            Some(ClearValue::Float(BACKGROUND_COLOR)),
            Some(ClearValue::Depth(1.0)),
        ];
        cb_builder
            .begin_render_pass(render_pass_begin_info, SubpassContents::Inline)
            .unwrap();

        // We set the viewport here
        let dimensions = framebuffer.extent().map(|i| i as f32);
        cb_builder.set_viewport(
            0,
            [Viewport {
                origin: [0.0, 0.0],
                dimensions,
                depth_range: 0.0..1.0,
            }],
        );

        // We need to generate a descriptor set pointing to a buffer with camera data in it
        let camera_data_buffer = self.camera.get_uniforms(&self.gfx_queue);
        let camera_descriptor_set = PersistentDescriptorSet::new(
            self.render_pipeline
                .pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [WriteDescriptorSet::buffer(0, camera_data_buffer)],
        )
        .unwrap();

        // Let's render the test model
        cb_builder
            // First we need to bind the mesh drawing pipeline
            .bind_pipeline_graphics(self.render_pipeline.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.render_pipeline.pipeline.layout().clone(),
                0,
                camera_descriptor_set,
            )
            // Then we bind the vertex and index buffers for the test model
            .bind_vertex_buffers(0, self.test_model.vertex_buffer.clone())
            .bind_index_buffer(self.test_model.index_buffer.clone())
            // And then we issue a draw_indexed call to render it to the framebuffer
            .draw_indexed(self.test_model.index_count, 1, 0, 0, 0)
            .unwrap();

        // Conclude the renderpass
        cb_builder.end_render_pass().unwrap();

        // Build command buffer
        let command_buffer = cb_builder.build().unwrap();

        // We need to ensure that the skybox texture has loaded before executing commands that reference it,
        // so we chain the future in if there is one. In a more structured engine, you'd do this somewhere else.
        let mut pre_exec_future = match self.skybox_upload_future.take() {
            Some(future) => before_future.join(future).boxed(),
            None => before_future,
        };
        // We'll also take the opportunity to clean up any finished futures
        pre_exec_future.cleanup_finished();

        // Execute after before_future and return the new future
        pre_exec_future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap()
            .boxed()
    }

    pub fn init_framebuffer(&mut self, image_view: SwapchainImageView, image_size: [u32; 2]) {
        // Create depth buffer
        let depth_buffer_image = AttachmentImage::transient(
            self.gfx_queue.device().clone(),
            image_size,
            vulkano::format::Format::D16_UNORM,
        )
        .unwrap();
        let depth_buffer_view = ImageView::new_default(depth_buffer_image).unwrap();

        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![image_view, depth_buffer_view],
                ..Default::default()
            },
        )
        .unwrap();

        self.framebuffer = Some(framebuffer);
    }

    /// Update times and dt at the end of each frame and move the camera
    pub fn update(&mut self) {
        // Each second, update average fps & reset frame count & dt sum
        if self.dt_sum > 1.0 {
            self.avg_fps = self.frame_count / self.dt_sum;
            self.frame_count = 0.0;
            self.dt_sum = 0.0;
        }
        self.dt = self.time.elapsed().as_secs_f32();
        self.dt_sum += self.dt;
        self.frame_count += 1.0;
        self.time = Instant::now();

        // Move the camera
        self.camera.phi += self.dt * CAMERA_ROTATION_SPEED;
    }

    /// Return average fps
    pub fn avg_fps(&self) -> f32 {
        self.avg_fps
    }

    /// Delta time in milliseconds
    pub fn dt(&self) -> f32 {
        self.dt * 1000.0
    }

    pub fn update_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.camera.set_aspect_ratio(aspect_ratio);
    }
}

/// A struct to hold parameters for the scene's camera.
/// It models a spherical camera that points inward at its `position`,
/// at a distance of `r`, and with `theta` and `phi` rotations.
struct Camera {
    /// The `[x, y, z]` center of the spherical camera
    center: Point3<f32>,

    /// The distance of the camera from its center
    r: f32,
    /// The rotation of the camera around its equator in degrees.
    theta: f32,
    /// The rotation of the camera between its poles, in degrees.
    phi: f32,

    /// The aspect ratio of the camera. It'll need to be set externally.
    pub aspect_ratio: f32,
}

impl Camera {
    pub fn new(center: Point3<f32>, r: f32, theta: f32, phi: f32) -> Self {
        Self {
            center,
            r,
            theta,
            phi,

            aspect_ratio: 1.0,
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    pub fn get_uniforms(&self, gfx_queue: &Arc<Queue>) -> Arc<CpuAccessibleBuffer<CameraData>> {
        let camera_data = CameraData {
            position: self.camera_position().into(),
            _dummy0: [0; 4],
            view: self.view_matrix().into(),
            proj: self.projection_matrix().into(),
        };

        CpuAccessibleBuffer::from_data(
            gfx_queue.device().clone(),
            BufferUsage::storage_buffer(),
            false,
            camera_data,
        )
        .unwrap()
    }

    fn camera_position(&self) -> Point3<f32> {
        // We'll convert our spherical coordinate angles to radians
        let theta_rad = Rad::from(Deg(self.theta));
        let phi_rad = Rad::from(Deg(self.phi));

        // And compute the cartesian equivalent positions
        let spherical_x = self.r * Rad::sin(phi_rad) * Rad::cos(theta_rad);
        let spherical_y = self.r * Rad::sin(phi_rad) * Rad::sin(theta_rad);
        let spherical_z = self.r * Rad::cos(phi_rad);

        let camera_position = [
            self.center.x + spherical_x,
            self.center.y + spherical_y,
            self.center.z + spherical_z,
        ];

        Point3::from(camera_position)
    }

    fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(self.camera_position(), self.center, UP_DIRECTION)
    }

    fn projection_matrix(&self) -> Matrix4<f32> {
        cgmath::perspective(
            Rad::from(Deg(FIELD_OF_VIEW_ANGLE)),
            self.aspect_ratio,
            NEAR_DISTANCE,
            FAR_DISTANCE,
        )
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            100.0, // The Utah Teapot is really big!
            45.0,
            45.0,
        )
    }
}

struct Skybox {
    pub cube_mesh: Model,
    pub cube_map: Arc<ImageView<ImmutableImage>>,
    pub cube_sampler: Arc<Sampler>,
}

impl Skybox {
    pub fn new(
        gfx_queue: &Arc<Queue>,
    ) -> (
        Self,
        CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>,
    ) {
        let cube_mesh = Model::cube(gfx_queue);
        // We need to load the data from each texture representing one of the six faces
        // of the cubemap. I think the order of data matters here, and I am basing it on
        // the table from https://www.khronos.org/opengl/wiki/Cubemap_Texture#Upload_and_orientation
        // TODO revise when you're sure
        let cube_map_data: Vec<u8> = [
            SKYBOX_RIGHT,  // POSITIVE_X
            SKYBOX_LEFT,   // NEGATIVE_X
            SKYBOX_TOP,    // POSITIVE_Y
            SKYBOX_BOTTOM, // NEGATIVE_Y
            SKYBOX_FRONT,  // POSITIVE_Z
            SKYBOX_BACK,   // NEGATIVE_Z
        ]
        .into_iter()
        // Note the use of flat_map, even though we have 6 images, they will be sent as a single sequence
        // of u8s. Vulkan will differentiate the images using the dimensions and format data we pass.
        .flat_map(|path| {
            // We load the image using the image crate
            let image = ImageReader::open(path).unwrap().decode().unwrap();
            // Then we get a vector containing width * height sets of rgb u8s
            let image_data = image.as_rgb8().unwrap().as_raw().to_owned();

            // On most graphics hardware, SRGB images aren't supported, they want SRGBA.
            // So we'll convert the RGB of the images to RGBA by pushing a maximum alpha
            // value every 3 u8s, i.e., every unit of rgb color data.
            let mut actual_data =
                Vec::<u8>::with_capacity((image.width() * image.height() * 4) as usize);
            for rgb_index in 0..(image_data.len() / 3) {
                actual_data.push(image_data[rgb_index]); // R
                actual_data.push(image_data[rgb_index + 1]); // G
                actual_data.push(image_data[rgb_index + 2]); // B
                actual_data.push(255); // A
            }

            // The image data is returned
            actual_data
        })
        .collect();

        // We'll reopen the first image to read its info
        let image = ImageReader::open(SKYBOX_RIGHT).unwrap().decode().unwrap();

        // This is uploaded to the GPU as an ImmutableImage, because we won't be writing to it at any point.
        // Internally, this function writes to a transfer buffer and then executes a command to move the transfer
        // data to the final image location, so it returns the ImmutableImage and a CommandBufferExecFuture
        // representing the moment that the image becomes available for use.
        let (cube_map_image, upload_future) = ImmutableImage::from_iter(
            cube_map_data,
            ImageDimensions::Dim2d {
                width: image.width(),
                height: image.height(),
                array_layers: 6, // 6 layers, one for each face
            },
            vulkano::image::MipmapsCount::One,
            vulkano::format::Format::R8G8B8A8_SRGB,
            gfx_queue.clone(),
        )
        .unwrap();

        // We create the cubemap image view, we need to fill the ImageViewCreateInfo
        // manually, as the default is made for standard textures.
        let cube_map = ImageView::new_default(cube_map_image).unwrap();
        // let cube_map = ImageView::new(
        //     cube_map_image,
        //     ImageViewCreateInfo {
        //         view_type: vulkano::image::view::ImageViewType::Cube,
        //         format: Some(vulkano::format::Format::R8G8B8_SRGB),
        //         ..Default::default()
        //     },
        // )
        // .unwrap();

        // We will also create a sampler for sampling the cubemap
        let cube_sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                ..Default::default()
            },
        )
        .unwrap();

        (
            Self {
                cube_mesh,
                cube_map,
                cube_sampler,
            },
            upload_future,
        )
    }
}

struct Model {
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    index_count: u32,
}

impl Model {
    pub fn teapot(gfx_queue: &Arc<Queue>) -> Self {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            gfx_queue.device().clone(),
            BufferUsage::vertex_buffer(),
            false,
            TEST_MODEL_VERTICES,
        )
        .unwrap();

        let index_buffer = CpuAccessibleBuffer::from_iter(
            gfx_queue.device().clone(),
            BufferUsage::index_buffer(),
            false,
            TEST_MODEL_INDICES.map(|i| i as u32),
        )
        .unwrap();

        Self {
            vertex_buffer,
            index_buffer,
            index_count: TEST_MODEL_INDICES.len() as u32,
        }
    }

    pub fn cube(gfx_queue: &Arc<Queue>) -> Self {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            gfx_queue.device().clone(),
            BufferUsage::vertex_buffer(),
            false,
            CUBE_MODEL_VERTICES,
        )
        .unwrap();

        let index_buffer = CpuAccessibleBuffer::from_iter(
            gfx_queue.device().clone(),
            BufferUsage::index_buffer(),
            false,
            CUBE_MODEL_INDICES.map(|i| i as u32),
        )
        .unwrap();

        Self {
            vertex_buffer,
            index_buffer,
            index_count: CUBE_MODEL_INDICES.len() as u32,
        }
    }
}

const CUBE_MODEL_VERTICES: [Vertex; 8] = [
    Vertex {
        position: [-0.5, -0.5, 0.5],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
    },
    Vertex {
        position: [-0.5, -0.5, -0.5],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
    },
];

const CUBE_MODEL_INDICES: [u32; 36] = [
    1, 2, 0, 3, 6, 2, 7, 4, 6, 5, 0, 4, 6, 0, 2, 3, 5, 7, 1, 3, 2, 3, 7, 6, 7, 5, 4, 5, 1, 0, 6, 4,
    0, 3, 1, 5,
];
