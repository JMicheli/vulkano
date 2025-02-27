// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example is a copy of `basic-compute-shaders.rs`, but initalizes half of the input buffer
// and then we use `copy_buffer_dimensions` to copy the first half of the input buffer to the second half.

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BufferCopy, CommandBufferUsage, CopyBufferInfoTyped,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

fn main() {
    let instance = Instance::new(InstanceCreateInfo {
        // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
        enumerate_portability: true,
        ..Default::default()
    })
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_compute())
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let pipeline = {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
                    #version 450

                    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer Data {
                        uint data[];
                    } data;

                    void main() {
                        uint idx = gl_GlobalInvocationID.x;
                        data.data[idx] *= 12;
                    }
                "
            }
        }
        let shader = cs::load(device.clone()).unwrap();
        ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap()
    };

    let data_buffer = {
        // we intitialize half of the array and leave the other half to 0, we will use copy later to fill it
        let data_iter = (0..65536u32).map(|n| if n < 65536 / 2 { n } else { 0 });
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                storage_buffer: true,
                transfer_src: true,
                transfer_dst: true,
                ..BufferUsage::none()
            },
            false,
            data_iter,
        )
        .unwrap()
    };

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        // copy from the first half to the second half (inside the same buffer) before we run the computation
        .copy_buffer(CopyBufferInfoTyped {
            regions: [BufferCopy {
                src_offset: 0,
                dst_offset: 65536 / 2,
                size: 65536 / 2,
                ..Default::default()
            }]
            .into(),
            ..CopyBufferInfoTyped::buffers(data_buffer.clone(), data_buffer.clone())
        })
        .unwrap()
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set.clone(),
        )
        .dispatch([1024, 1, 1])
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let data_buffer_content = data_buffer.read().unwrap();

    // here we have the same data in the two halfs of the buffer
    for n in 0..65536 / 2 {
        // the two halfs should have the same data
        assert_eq!(data_buffer_content[n as usize], n * 12);
        assert_eq!(data_buffer_content[n as usize + 65536 / 2], n * 12);
    }
}
