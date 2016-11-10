// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error::Error;
use std::iter;
use std::sync::Arc;

use buffer::TrackedBuffer;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::DynamicState;
use command_buffer::pool::CommandPool;
use command_buffer::StatesManager;
use command_buffer::SubmitInfo;
use command_buffer::SubmitBuilder;
use command_buffer::Submit;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use descriptor::PipelineLayoutRef;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use device::Device;
use device::Queue;
use framebuffer::traits::TrackedFramebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use image::Layout;
use image::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::vertex::Source;
use sync::Fence;
use vk;

pub use self::begin_render_pass::CmdBeginRenderPass;
pub use self::bind_index_buffer::CmdBindIndexBuffer;
pub use self::bind_descriptor_sets::{CmdBindDescriptorSets, CmdBindDescriptorSetsError};
pub use self::bind_pipeline::CmdBindPipeline;
pub use self::bind_vertex_buffers::CmdBindVertexBuffers;
pub use self::blit_image_unsynced::{BlitRegion, BlitRegionAspect};
pub use self::blit_image_unsynced::{CmdBlitImageUnsynced, CmdBlitImageUnsyncedError};
pub use self::copy_buffer::{CmdCopyBuffer, CmdCopyBufferError};
pub use self::draw::CmdDraw;
pub use self::empty::{empty, EmptyCommandsList};
pub use self::end_render_pass::CmdEndRenderPass;
pub use self::fill_buffer::{CmdFillBuffer, CmdFillBufferError};
pub use self::next_subpass::CmdNextSubpass;
pub use self::push_constants::{CmdPushConstants, CmdPushConstantsError};
pub use self::set_state::{CmdSetState};
pub use self::update_buffer::{CmdUpdateBuffer, CmdUpdateBufferError};

mod begin_render_pass;
mod bind_descriptor_sets;
mod bind_index_buffer;
mod bind_pipeline;
mod bind_vertex_buffers;
mod blit_image_unsynced;
mod copy_buffer;
pub mod dispatch;
mod draw;
mod empty;
mod end_render_pass;
pub mod execute;
mod fill_buffer;
mod next_subpass;
mod push_constants;
mod set_state;
mod update_buffer;

/// A list of commands that can be turned into a command buffer.
///
/// This is just a naked list of commands. It stores buffers, images, etc. but the list of commands
/// itself is not a Vulkan object.
pub unsafe trait CommandsList {
    /// Adds a command that writes the content of a buffer.
    ///
    /// After this command is executed, the content of `buffer` will become `data`.
    #[inline]
    fn update_buffer<'a, B, D: ?Sized>(self, buffer: B, data: &'a D)
                                       -> Result<CmdUpdateBuffer<'a, Self, B, D>, CmdUpdateBufferError>
        where Self: Sized + CommandsListPossibleOutsideRenderPass, B: TrackedBuffer, D: Copy + 'static
    {
        CmdUpdateBuffer::new(self, buffer, data)
    }

    /// Adds a command that copies the content of a buffer to another.
    #[inline]
    fn copy_buffer<S, D>(self, source: S, destination: D)
                         -> Result<CmdCopyBuffer<Self, S, D>, CmdCopyBufferError>
        where Self: Sized + CommandsListPossibleOutsideRenderPass,
              S: TrackedBuffer, D: TrackedBuffer
    {
        CmdCopyBuffer::new(self, source, destination)
    }

    /// Adds a command that writes the content of a buffer.
    #[inline]
    fn fill_buffer<B>(self, buffer: B, data: u32)
                      -> Result<CmdFillBuffer<Self, B>, CmdFillBufferError>
        where Self: Sized + CommandsListPossibleOutsideRenderPass, B: TrackedBuffer
    {
        CmdFillBuffer::new(self, buffer, data)
    }

    /// Adds a command that executes a secondary command buffer.
    ///
    /// When you create a command buffer, you have the possibility to create either a primary
    /// command buffer or a secondary command buffer. Secondary command buffers can't be executed
    /// directly, but can be executed from a primary command buffer.
    ///
    /// A secondary command buffer can't execute another secondary command buffer. The only way
    /// you can use `execute` is to make a primary command buffer call a secondary command buffer.
    #[inline]
    fn execute<Cb>(self, command_buffer: Cb) -> execute::ExecuteCommand<Cb, Self>
        where Self: Sized, Cb: CommandsListOutput       /* FIXME: */
    {
        execute::ExecuteCommand::new(self, command_buffer)
    }

    /// Adds a command that executes a compute shader.
    ///
    /// The `dimensions` are the number of working groups to start. The GPU will execute the
    /// compute shader `dimensions[0] * dimensions[1] * dimensions[2]` times.
    ///
    /// The `pipeline` is the compute pipeline that will be executed, and the sets and push
    /// constants will be accessible to all the invocations.
    #[inline]
    fn dispatch<'a, Pl, S, Pc>(self, pipeline: Arc<ComputePipeline<Pl>>, sets: S,
                               dimensions: [u32; 3], push_constants: &'a Pc)
                               -> dispatch::DispatchCommand<'a, Self, Pl, S, Pc>
        where Self: Sized + CommandsList + CommandsListPossibleOutsideRenderPass, Pl: PipelineLayoutRef,
              S: TrackedDescriptorSetsCollection, Pc: 'a
    {
        dispatch::DispatchCommand::new(self, pipeline, sets, dimensions, push_constants)
    }

    /// Adds a command that starts a render pass.
    ///
    /// If `secondary` is true, then you will only be able to add secondary command buffers while
    /// you're inside the first subpass on the render pass. If `secondary` is false, you will only
    /// be able to add inline draw commands and not secondary command buffers.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    fn begin_render_pass<F, C>(self, framebuffer: F, secondary: bool, clear_values: C)
                               -> CmdBeginRenderPass<Self, F::RenderPass, F>
        where Self: Sized, F: TrackedFramebuffer,
              F::RenderPass: RenderPass + RenderPassClearValues<C>
    {
        CmdBeginRenderPass::new(self, framebuffer, secondary, clear_values)
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    fn next_subpass(self, secondary: bool) -> CmdNextSubpass<Self>
        where Self: Sized
    {
        CmdNextSubpass::new(self, secondary)
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    fn end_render_pass(self) -> CmdEndRenderPass<Self>
        where Self: Sized
    {
        CmdEndRenderPass::new(self)
    }

    /// Adds a command that draws.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw<Pv, Pl, Prp, S, Pc, V>(self, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
                                   dynamic: DynamicState, vertices: V, sets: S,
                                   push_constants: Pc)
                                   -> CmdDraw<Self, V, Pv, Pl, Prp, S, Pc>
        where Self: Sized + CommandsList + CommandsListPossibleInsideRenderPass, Pl: PipelineLayoutRef,
              S: TrackedDescriptorSetsCollection, Pv: Source<V>
    {
        CmdDraw::new(self, pipeline, dynamic, vertices, sets, push_constants)
    }

    /// Appends this list of commands at the end of a command buffer in construction.
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        // TODO: temporary body until all the impls implement this function
        unimplemented!()
    }

    /// Returns true if the command buffer can be built. This function should always return true,
    /// except when we're building a primary command buffer that is inside a render pass.
    // TODO: remove function
    #[deprecated]
    fn buildable_state(&self) -> bool { unimplemented!() }

    /// Returns the number of commands in the commands list.
    ///
    /// Note that multiple actual commands may count for just 1.
    // TODO: remove function
    #[deprecated]
    fn num_commands(&self) -> usize { unimplemented!() }

    /// Checks whether the command can be executed on the given queue family.
    // TODO: error type?
    // TODO: remove function
    #[deprecated]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> { unimplemented!() }

    /// Extracts the object that contains the states of all the resources of the commands list.
    ///
    /// Panics if the states were already extracted.
    // TODO: remove function
    #[deprecated]
    fn extract_states(&mut self) -> StatesManager { unimplemented!() }

    /// Returns true if the given compute pipeline is currently binded in the commands list.
    // TODO: better API?
    // TODO: remove function
    #[deprecated]
    fn is_compute_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool { unimplemented!() }

    /// Returns true if the given graphics pipeline is currently binded in the commands list.
    // TODO: better API?
    // TODO: remove function
    #[deprecated]
    fn is_graphics_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool { unimplemented!() }
}

unsafe impl CommandsList for Box<CommandsList> {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        (**self).append(builder)
    }
}

/// Output of the "append" method. The lifetime corresponds to the CommandsList.
pub trait CommandsListSink<'a> {
    fn device(&self) -> &Arc<Device>;

    /// Note that the lifetime means that we hold a reference to the content of
    /// the commands list in that closure.
    fn add_command(&mut self, Box<CommandsListSinkCaller<'a> + 'a>);

    fn add_buffer_transition(&mut self, buffer: &TrackedBuffer, offset: usize, size: usize,
                             write: bool);

    ///
    ///
    /// If necessary, you must transition the image to the `layout`.
    fn add_image_transition(&mut self, image: &TrackedImage, first_layer: u32, num_layers: u32,
                            first_mipmap: u32, num_mipmaps: u32, layout: Layout);

    fn add_image_transition_notification(&mut self, image: &TrackedImage, first_layer: u32,
                                         num_layers: u32, first_mipmap: u32, num_mipmaps: u32,
                                         layout: Layout);
}

pub trait CommandsListSinkCaller<'a> {
    fn call(self: Box<Self>, &mut RawCommandBufferPrototype<'a>);
}
impl<'a, T> CommandsListSinkCaller<'a> for T where T: FnOnce(&mut RawCommandBufferPrototype<'a>) -> () + 'a {
    fn call(self: Box<Self>, proto: &mut RawCommandBufferPrototype<'a>) {
        self(proto);
    }
}

#[deprecated]
pub unsafe trait CommandsListConcrete: CommandsList {
    type Pool: CommandPool;
    /// The type of the command buffer that will be generated.
    type Output: CommandsListOutput;

    /// Turns the commands list into a command buffer.
    ///
    /// This function accepts additional arguments that will customize the output:
    ///
    /// - `additional_elements` is a closure that must be called on the command buffer builder
    ///   after it has finished building and before `final_barrier` are added.
    /// - `barriers` is a list of pipeline barriers accompanied by a command number. The
    ///   pipeline barrier must happen after the given command number. Usually you want all the
    ///   the command numbers to be inferior to `num_commands`.
    /// - `final_barrier` is a pipeline barrier that must be added at the end of the
    ///   command buffer builder.
    ///
    /// This function doesn't check that `buildable_state` returns true.
    #[deprecated]
    unsafe fn raw_build<I, F>(self, in_s: &mut StatesManager, out: &mut StatesManager,
                              additional_elements: F, barriers: I,
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<Self::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>;

    /// Turns the commands list into a command buffer that can be submitted.
    // This function isn't inline because `raw_build` implementations usually are inline.
    #[deprecated]
    fn build(mut self) -> CommandBuffer<Self::Output> where Self: Sized {
        assert!(self.buildable_state(), "Tried to build a command buffer still inside a \
                                         render pass");

        let mut states_in = self.extract_states();
        let mut states_out = StatesManager::new(); 

        let output = unsafe {
            self.raw_build(&mut states_in, &mut states_out, |_| {},
                           iter::empty(), PipelineBarrierBuilder::new())
        };

        CommandBuffer {
            states: states_out,
            commands: output,
        }
    }
}

/// Extension trait for both `CommandsList` and `CommandsListOutput` that indicates that we're
/// possibly outside a render pass.
///
/// In other words, if this trait is *not* implemented then we're guaranteed *not* to be outside
/// of a render pass. If it is implemented, then we maybe are but that's not sure.
pub unsafe trait CommandsListPossibleOutsideRenderPass {
    /// Returns `true` if we're outside a render pass.
    fn is_outside_render_pass(&self) -> bool;
}

/// Extension trait for both `CommandsList` and `CommandsListOutput` that indicates that we're
/// possibly inside a render pass.
///
/// In other words, if this trait is *not* implemented then we're guaranteed *not* to be inside
/// a render pass. If it is implemented, then we maybe are but that's not sure.
// TODO: make all return values optional, since we're possibly not in a render pass
pub unsafe trait CommandsListPossibleInsideRenderPass {
    type RenderPass: RenderPass;

    /// Returns the number of the subpass we're in. The value is 0-indexed, so immediately after
    /// calling `begin_render_pass` the value will be `0`.
    ///
    /// The value should always be strictly inferior to the number of subpasses in the render pass.
    fn current_subpass_num(&self) -> u32;

    /// If true, only secondary command buffers can be added inside the subpass. If false, only
    /// inline draw commands can be added.
    fn secondary_subpass(&self) -> bool;

    /// Returns the description of the render pass we're in.
    // TODO: return a trait object instead?
    fn render_pass(&self) -> &Self::RenderPass;

    //fn current_subpass(&self) -> Subpass<&Self::RenderPass>;
}

#[deprecated]
pub unsafe trait CommandsListOutput<S = StatesManager> {
    /// Returns the inner object.
    // TODO: crappy API
    #[deprecated]
    fn inner(&self) -> vk::CommandBuffer;

    /// Returns the device this object belongs to.
    #[deprecated]
    fn device(&self) -> &Arc<Device>;

    #[deprecated]
    unsafe fn on_submit(&self, states: &S, queue: &Arc<Queue>,
                        fence: &mut FnMut() -> Arc<Fence>) -> SubmitInfo;
}

#[deprecated]
pub struct CommandBuffer<C = Box<CommandsListOutput>> {
    states: StatesManager,
    commands: C,
}

unsafe impl<C> Submit for CommandBuffer<C> where C: CommandsListOutput {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.commands.device()
    }

    #[inline]
    unsafe fn append_submission<'a>(&'a self, mut base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        let mut fence = None;
        let infos = self.commands.on_submit(&self.states, queue, &mut || {
            match &mut fence {
                f @ &mut None => {
                    let fe = Fence::new(self.device().clone()); *f = Some(fe.clone()); fe
                },
                &mut Some(ref f) => f.clone()
            }
        });

        for (sem_wait, sem_stage) in infos.semaphores_wait {
            base = base.add_wait_semaphore(sem_wait, sem_stage);
        }

        if !infos.pre_pipeline_barrier.is_empty() {
            unimplemented!()
        }

        base = base.add_command_buffer_raw(self.commands.inner());

        if !infos.post_pipeline_barrier.is_empty() {
            unimplemented!()
        }

        for sem_signal in infos.semaphores_signal {
            base = base.add_signal_semaphore(sem_signal);
        }

        if let Some(fence) = fence {
            base = base.add_fence_signal(fence);
        }

        Ok(base)
    }
}
