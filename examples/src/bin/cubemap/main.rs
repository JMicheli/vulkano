// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

mod app;
mod render_pipeline;
mod shaders;

use crate::app::CubeMapApp;
use vulkano::swapchain::PresentMode;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

// TODO rewrite
/// This is an example demonstrating an application with some more non-trivial functionality.
/// It should get you more up to speed with how you can use Vulkano.
/// It contains
/// - Compute pipeline to calculate Mandelbrot and Julia fractals writing them to an image target
/// - Graphics pipeline to draw the fractal image over a quad that covers the whole screen
/// - Renderpass rendering that image over swapchain image
/// - An organized Renderer with functionality good enough to copy to other projects
/// - Simple FractalApp to handle runtime state
/// - Simple Input system to interact with the application
fn main() {
    // Create event loop
    let mut event_loop = EventLoop::new();
    let context = VulkanoContext::new(VulkanoConfig::default());
    let mut windows = VulkanoWindows::default();
    let _id = windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor {
            title: "Cubemap (loading textures)".to_string(),
            present_mode: PresentMode::Immediate,
            ..Default::default()
        },
        |_| {},
    );

    let primary_window_renderer = windows.get_primary_renderer_mut().unwrap();
    // Make sure the image usage is correct (based on your pipeline).

    // Create an app to hold resources we upload to the GPU. This example is non-interactive,
    // see the interactive-fractal example for input handling.
    let gfx_queue = context.graphics_queue();
    let mut app = CubeMapApp::new(gfx_queue, primary_window_renderer.swapchain_format());

    // TODO rewrite
    // Basic loop for our runtime
    // 1. Handle events
    // 2. Handle resizing
    // 3. Render the scene
    loop {
        if !handle_events(&mut event_loop, primary_window_renderer) {
            break;
        }

        match primary_window_renderer.window_size() {
            [w, h] => {
                // Skip this frame when minimized
                if w == 0.0 || h == 0.0 {
                    continue;
                }
            }
        }

        app.update_aspect_ratio(primary_window_renderer.aspect_ratio());
        render(primary_window_renderer, &mut app);
        app.update();
        primary_window_renderer
            .window()
            .set_title(&format!("Cubemap - fps: {:.2}", app.avg_fps(),));
    }
}

/// Handle events and return `bool` if we should quit
fn handle_events(event_loop: &mut EventLoop<()>, renderer: &mut VulkanoWindowRenderer) -> bool {
    let mut is_running = true;
    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match &event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => is_running = false,
                WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                    renderer.resize()
                }
                _ => (),
            },
            Event::MainEventsCleared => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
    is_running
}

/// Orchestrate rendering here
fn render(renderer: &mut VulkanoWindowRenderer, app: &mut CubeMapApp) {
    // Start frame
    let before_pipeline_future = match renderer.acquire() {
        Err(e) => {
            println!("{}", e.to_string());
            return;
        }
        Ok(future) => future,
    };

    // Set up framebuffer
    app.init_framebuffer(
        renderer.swapchain_image_view(),
        renderer.swapchain_image_size(),
    );
    // Input previous future. Draw scene to framebuffer
    let after_render_future = app.render(before_pipeline_future);
    // Finish frame (which presents the view). Input last future. Wait for the future so resources are not in use
    // when we render
    renderer.present(after_render_future, true);
}
