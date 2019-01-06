extern crate env_logger;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;
extern crate glsl_to_spirv;
extern crate winit;
extern crate rand;

use hal::{
    command, pool, pso, queue, Adapter, Backend,
    Capability, Device, Gpu, Compute, Instance, PhysicalDevice, QueueFamily, Features, memory, buffer, DescriptorPool,
};
use std::{mem, ptr};

// dumb SPIRV bytes
static RESERVED_ID: i32 = 0;
static FUNC_ID: i32 = 1;
static IN_ID: i32 = 2;
static OUT_ID: i32 = 3;
static GLOBAL_INVOCATION_ID: i32 = 4;
static VOID_TYPE_ID: i32 = 5;
static FUNC_TYPE_ID: i32 = 6;
static INT_TYPE_ID: i32 = 7;
static INT_ARRAY_TYPE_ID: i32 = 8;
static STRUCT_ID: i32 = 9;
static POINTER_TYPE_ID: i32 = 10;
static ELEMENT_POINTER_TYPE_ID: i32 = 11;
static INT_VECTOR_TYPE_ID: i32 = 12;
static INT_VECTOR_POINTER_TYPE_ID: i32 = 13;
static INT_POINTER_TYPE_ID: i32 = 14;
static CONSTANT_ZERO_ID: i32 = 15;
static CONSTANT_ARRAY_LENGTH_ID: i32 = 16;
static LABEL_ID: i32 = 17;
static IN_ELEMENT_ID: i32 = 18;
static OUT_ELEMENT_ID: i32 = 19;
static GLOBAL_INVOCATION_X_ID: i32 =20;
static GLOBAL_INVOCATION_X_PTR_ID: i32 =21;
static TEMP_LOADED_ID: i32 = 22;
static BOUND: i32 = 23;
static INPUT: i32 = 1;
static UNIFORM: i32 = 2;
static BUFFER_BLOCK: i32 = 3;
static ARRAY_STRIDE: i32 = 6;
static BUILTIN: i32 = 11;
static BINDING: i32 = 33;
static OFFSET: i32 = 35;
static DESCRIPTOR_SET: i32 = 34;
static GLOBAL_INVOCATION: i32 = 28;
static OP_TYPE_VOID: i32 = 19;
static OP_TYPE_FUNCTION: i32 = 33;
static OP_TYPE_INT: i32 = 21;
static OP_TYPE_VECTOR: i32 = 23;
static OP_TYPE_ARRAY: i32 = 28;
static OP_TYPE_STRUCT: i32 = 30;
static OP_TYPE_POINTER: i32 = 32;
static OP_VARIABLE: i32 = 59;
static OP_DECORATE: i32 = 71;
static OP_MEMBER_DECORATE: i32 = 72;
static OP_FUNCTION: i32 = 54;
static OP_LABEL: i32 = 248;
static OP_ACCESS_CHAIN: i32 = 65;
static OP_CONSTANT: i32 = 43;
static OP_LOAD: i32 = 61;
static OP_STORE: i32 = 62;
static OP_RETURN: i32 = 253;
static OP_FUNCTION_END: i32 = 56;
static OP_CAPABILITY: i32 = 17;
static OP_MEMORY_MODEL: i32 = 14;
static OP_ENTRY_POINT: i32 = 15;
static OP_EXECUTION_MODE: i32 = 16;
static OP_COMPOSITE_EXTRACT: i32 = 81;

// from OmniViral (Zakarum)
pub fn cast_vec<T>(mut vec: Vec<T>) -> Vec<u8> {
    let raw_len = mem::size_of::<T>()*vec.len();
    let len = raw_len;

    let cap = mem::size_of::<T>()*vec.capacity();

    let ptr = vec.as_mut_ptr();
    mem::forget(vec);
    unsafe { Vec::from_raw_parts(ptr as _, len, cap) }
}


fn main() {
    env_logger::init();
    unsafe {
        let mut application = ComputeApplication::init();
        application.execute_compute();
        application.check_result();
        application.clean_up();
    }
}

#[derive(Default)]
struct QueueFamilyIds {
    compute_family: Option<queue::QueueFamilyId>,
}

impl QueueFamilyIds {
    fn is_complete(&self) -> bool {
        self.compute_family.is_some()
    }
}

struct ComputeApplication {
    command_buffer: command::CommandBuffer<back::Backend, Compute, command::OneShot, command::Primary>,
    command_pool: pool::CommandPool<back::Backend, Compute>,
    descriptor_sets: Vec<<back::Backend as Backend>::DescriptorSet>,
    compute_pipeline: <back::Backend as Backend>::ComputePipeline,
    descriptor_set_layout: <back::Backend as Backend>::DescriptorSetLayout,
    pipeline_layout: <back::Backend as Backend>::PipelineLayout,
    result: *mut u8,
    out_buffer: <back::Backend as Backend>::Buffer,
    payload: *mut u8,
    in_buffer: <back::Backend as Backend>::Buffer,
    buffer_size: u32,
    buffer_length: u32,
    command_queues: Vec<queue::CommandQueue<back::Backend, Compute>>,
    device: <back::Backend as Backend>::Device,
    _adapter: Adapter<back::Backend>,
    _instance: back::Instance,
}

impl ComputeApplication {
    unsafe fn init() -> ComputeApplication {
        let instance = ComputeApplication::create_instance();
        let mut adapter = ComputeApplication::pick_adapter(&instance);
        let (device, command_queues, queue_type, qf_id) =
            ComputeApplication::create_device_with_compute_queue(&mut adapter);

        let (buffer_length, buffer_size, in_buffer, payload, out_buffer, result) = ComputeApplication::create_io_buffers(&mut adapter, &device);
        let (descriptor_set_layout, pipeline_layout, compute_pipeline) =
            ComputeApplication::create_compute_pipeline(buffer_length, &device);

        let descriptor_sets = ComputeApplication::set_up_descriptor_sets(&device, &descriptor_set_layout, buffer_size, &in_buffer, &out_buffer);

        let mut command_pool =
            ComputeApplication::create_command_pool(&device, queue_type, qf_id);
        let command_buffer = ComputeApplication::create_command_buffer(
            buffer_size,
            &mut command_pool,
            &descriptor_sets,
            &pipeline_layout,
            &compute_pipeline,
        );

        ComputeApplication {
            command_buffer,
            command_pool,
            descriptor_sets,
            compute_pipeline,
            descriptor_set_layout,
            pipeline_layout,
            result,
            out_buffer,
            payload,
            in_buffer,
            buffer_size,
            buffer_length,
            command_queues,
            device,
            _adapter: adapter,
            _instance: instance,
        }
    }

    fn create_instance() -> back::Instance {
        back::Instance::create("compute-example", 1)
    }

    fn find_queue_families(adapter: &Adapter<back::Backend>) -> QueueFamilyIds {
        let mut queue_family_ids = QueueFamilyIds::default();

        for queue_family in &adapter.queue_families {
            if queue_family.max_queues() > 0 && queue_family.supports_graphics() {
                queue_family_ids.compute_family = Some(queue_family.id());
            }

            if queue_family_ids.is_complete() {
                break;
            }
        }

        queue_family_ids
    }

    fn is_adapter_suitable(adapter: &Adapter<back::Backend>) -> bool {
        ComputeApplication::find_queue_families(adapter).is_complete()
    }

    fn pick_adapter(instance: &back::Instance) -> Adapter<back::Backend> {
        let adapters = instance.enumerate_adapters();
        for adapter in adapters {
            if ComputeApplication::is_adapter_suitable(&adapter) {
                return adapter;
            }
        }
        panic!("No suitable adapter");
    }

    fn create_device_with_compute_queue(
        adapter: &mut Adapter<back::Backend>,
    ) -> (
        <back::Backend as Backend>::Device,
        Vec<queue::CommandQueue<back::Backend, Compute>>,
        queue::QueueType,
        queue::family::QueueFamilyId,
    ) {
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                Compute::supported_by(family.queue_type())
                    && family.max_queues() > 0
            })
            .expect("Could not find a queue family supporting graphics.");

        let priorities = vec![1.0; 1];
        let families = [(family, priorities.as_slice())];

        let Gpu { device, mut queues } = unsafe {
            adapter
                .physical_device
                .open(&families, Features::empty())
                .expect("Could not create device.")
        };

        let mut queue_group = queues
            .take::<Compute>(family.id())
            .expect("Could not take ownership of relevant queue group.");

        let command_queues: Vec<_> = queue_group.queues.drain(..1).collect();

        (device, command_queues, family.queue_type(), family.id())
    }

    unsafe fn create_io_buffers(adapter: &mut Adapter<back::Backend>, device: &<back::Backend as Backend>::Device)
        -> (u32, u32, <back::Backend as Backend>::Buffer, *mut u8, <back::Backend as Backend>::Buffer, *mut u8)
    {

        let buffer_length: u32 = 16384;
        let buffer_size: u32 = (mem::size_of::<i32>() as u32)*buffer_length;
        let required_size: u64 = 2*(buffer_size as u64);

        let mut in_buffer = device.create_buffer(buffer_size as u64, buffer::Usage::STORAGE).unwrap();
        let mut out_buffer = device.create_buffer(buffer_size as u64, buffer::Usage::STORAGE).unwrap();

        let memory_properties = adapter.physical_device.memory_properties();

        let memory_type_id: hal::MemoryTypeId = memory_properties.memory_types
            .iter()
            .position(|mt| {
                mt
                    .properties
                    .contains(memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT) &&
                required_size <= memory_properties.memory_heaps[mt.heap_index]
            })
            .unwrap()
            .into();

        let memory = device.allocate_memory(memory_type_id, required_size as u64).unwrap();

        device.bind_buffer_memory(&memory, 0, &mut in_buffer).unwrap();
        device.bind_buffer_memory(&memory, buffer_size as u64, &mut out_buffer).unwrap();

        let payload = device.map_memory(&memory, (0 as u64)..(buffer_size as u64)).unwrap();
        let result = device.map_memory(&memory, (buffer_size as u64)..(required_size as u64)).unwrap();

        (buffer_length, buffer_size, in_buffer, payload, out_buffer, result)
    }

    unsafe fn create_compute_pipeline(buffer_length: u32,
        device: &<back::Backend as Backend>::Device
    ) -> (
        <back::Backend as Backend>::DescriptorSetLayout,
        <back::Backend as Backend>::PipelineLayout,
        <back::Backend as Backend>::ComputePipeline,
    ) {
        // eldritch magic
        let spirv_i32: Vec<i32> = vec![
            // first is the SPIR-V header
            0x07230203 as i32, // magic header ID
            0x00010000 as i32, // version 1.0.0
            0,          // generator (optional)
            BOUND,      // bound
            0,          // schema

            // OpCapability Shader
            (2 << 16) | OP_CAPABILITY, 1,

            // OpMemoryModel Logical Simple
            (3 << 16) | OP_MEMORY_MODEL, 0, 0,

            // OpEntryPoint GLCompute %FUNC_ID "f" %IN_ID %OUT_ID
            (4 << 16) | OP_ENTRY_POINT, 5, FUNC_ID, 0x00000066 as i32,

            // OpExecutionMode %FUNC_ID LocalSize 1 1 1
            (6 << 16) | OP_EXECUTION_MODE, FUNC_ID, 17, 1, 1, 1,

            // next declare decorations

            (3 << 16) | OP_DECORATE, STRUCT_ID, BUFFER_BLOCK,

            (4 << 16) | OP_DECORATE, GLOBAL_INVOCATION_ID, BUILTIN, GLOBAL_INVOCATION,

            (4 << 16) | OP_DECORATE, IN_ID, DESCRIPTOR_SET, 0,

            (4 << 16) | OP_DECORATE, IN_ID, BINDING, 0,

            (4 << 16) | OP_DECORATE, OUT_ID, DESCRIPTOR_SET, 0,

            (4 << 16) | OP_DECORATE, OUT_ID, BINDING, 1,

            (4 << 16) | OP_DECORATE, INT_ARRAY_TYPE_ID, ARRAY_STRIDE, 4,

            (5 << 16) | OP_MEMBER_DECORATE, STRUCT_ID, 0, OFFSET, 0,

            // next declare types
            (2 << 16) | OP_TYPE_VOID, VOID_TYPE_ID,

            (3 << 16) | OP_TYPE_FUNCTION, FUNC_TYPE_ID, VOID_TYPE_ID,

            (4 << 16) | OP_TYPE_INT, INT_TYPE_ID, 32, 1,

            (4 << 16) | OP_CONSTANT, INT_TYPE_ID, CONSTANT_ARRAY_LENGTH_ID, buffer_length as i32,

            (4 << 16) | OP_TYPE_ARRAY, INT_ARRAY_TYPE_ID, INT_TYPE_ID, CONSTANT_ARRAY_LENGTH_ID,

            (3 << 16) | OP_TYPE_STRUCT, STRUCT_ID, INT_ARRAY_TYPE_ID,

            (4 << 16) | OP_TYPE_POINTER, POINTER_TYPE_ID, UNIFORM, STRUCT_ID,

            (4 << 16) | OP_TYPE_POINTER, ELEMENT_POINTER_TYPE_ID, UNIFORM, INT_TYPE_ID,

            (4 << 16) | OP_TYPE_VECTOR, INT_VECTOR_TYPE_ID, INT_TYPE_ID, 3,

            (4 << 16) | OP_TYPE_POINTER, INT_VECTOR_POINTER_TYPE_ID, INPUT, INT_VECTOR_TYPE_ID,

            (4 << 16) | OP_TYPE_POINTER, INT_POINTER_TYPE_ID, INPUT, INT_TYPE_ID,

            // then declare constants
            (4 << 16) | OP_CONSTANT, INT_TYPE_ID, CONSTANT_ZERO_ID, 0,

            // then declare variables
            (4 << 16) | OP_VARIABLE, POINTER_TYPE_ID, IN_ID, UNIFORM,

            (4 << 16) | OP_VARIABLE, POINTER_TYPE_ID, OUT_ID, UNIFORM,

            (4 << 16) | OP_VARIABLE, INT_VECTOR_POINTER_TYPE_ID, GLOBAL_INVOCATION_ID, INPUT,

            // then declare function
            (5 << 16) | OP_FUNCTION, VOID_TYPE_ID, FUNC_ID, 0, FUNC_TYPE_ID,

            (2 << 16) | OP_LABEL, LABEL_ID,

            (5 << 16) | OP_ACCESS_CHAIN, INT_POINTER_TYPE_ID, GLOBAL_INVOCATION_X_PTR_ID, GLOBAL_INVOCATION_ID, CONSTANT_ZERO_ID,

            (4 << 16) | OP_LOAD, INT_TYPE_ID, GLOBAL_INVOCATION_X_ID, GLOBAL_INVOCATION_X_PTR_ID,

            (6 << 16) | OP_ACCESS_CHAIN, ELEMENT_POINTER_TYPE_ID, IN_ELEMENT_ID, IN_ID, CONSTANT_ZERO_ID, GLOBAL_INVOCATION_X_ID,

            (4 << 16) | OP_LOAD, INT_TYPE_ID, TEMP_LOADED_ID, IN_ELEMENT_ID,

            (6 << 16) | OP_ACCESS_CHAIN, ELEMENT_POINTER_TYPE_ID, OUT_ELEMENT_ID, OUT_ID, CONSTANT_ZERO_ID, GLOBAL_INVOCATION_X_ID,

            (3 << 16) | OP_STORE, OUT_ELEMENT_ID, TEMP_LOADED_ID,

            (1 << 16) | OP_RETURN,

            (1 << 16) | OP_FUNCTION_END,
        ];

        let shader_code: Vec<u8> = cast_vec(spirv_i32);

        let shader_module = device
            .create_shader_module(&shader_code)
            .expect("Error creating shader module.");

        let descriptor_set_layout_bindings: Vec<pso::DescriptorSetLayoutBinding> = vec![
            pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,
                immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 1,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,
                immutable_samplers: false,
            }
        ];

        let descriptor_set_layout = device.create_descriptor_set_layout(descriptor_set_layout_bindings, &[]).unwrap();
        let pipeline_layout = device.create_pipeline_layout(vec![&descriptor_set_layout], &[]).unwrap();

        let shader_entry_point = pso::EntryPoint {
            entry: "f",
            module: &shader_module,
            specialization: pso::Specialization {
                constants: &[],
                data: &[],
            }
        };

        let compute_pipeline_desc = pso::ComputePipelineDesc {
            shader: shader_entry_point,
            layout: &pipeline_layout,
            flags: pso::PipelineCreationFlags::empty(),
            parent: pso::BasePipeline::None,
        };

        let compute_pipeline = device.create_compute_pipeline(&compute_pipeline_desc, None).unwrap();

        device.destroy_shader_module(shader_module);
        (descriptor_set_layout, pipeline_layout, compute_pipeline)
    }

    unsafe fn create_command_pool(
        device: &<back::Backend as Backend>::Device,
        queue_type: queue::QueueType,
        qf_id: queue::family::QueueFamilyId,
    ) -> pool::CommandPool<back::Backend, Compute> {
        let raw_command_pool = device
            .create_command_pool(qf_id, pool::CommandPoolCreateFlags::empty())
            .unwrap();

        // safety check necessary before creating a strongly typed command pool
        assert_eq!(Compute::supported_by(queue_type), true);
        pool::CommandPool::new(raw_command_pool)
    }

    unsafe fn set_up_descriptor_sets(
        device: &<back::Backend as Backend>::Device,
        descriptor_set_layout: &<back::Backend as Backend>::DescriptorSetLayout,
        buffer_size: u32,
        in_buffer: &<back::Backend as Backend>::Buffer,
        out_buffer: &<back::Backend as Backend>::Buffer,
    ) -> Vec<<back::Backend as Backend>::DescriptorSet> {
        let descriptor_pool_size = pso::DescriptorRangeDesc {
            ty: pso::DescriptorType::StorageBuffer,
            count: 2,
        };

        let descriptor_pool = device.create_descriptor_pool(1, vec![&descriptor_pool_size]).unwrap();

        let descriptor_set = descriptor_pool.allocate_set(descriptor_set_layout).unwrap();

        let in_descriptor = hal::pso::Descriptor::Buffer(in_buffer, Some(0 as u64)..Some(buffer_size as u64));

        let out_descriptor = hal::pso::Descriptor::Buffer(out_buffer, Some(buffer_size as u64)..Some(2*buffer_size as u64));

        // how to know that I should be using Some(descriptor) here, based on docs?
        let in_descriptor_set_write = hal::pso::DescriptorSetWrite {
            set: &descriptor_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(in_descriptor),
        };

        let out_descriptor_set_write = hal::pso::DescriptorSetWrite {
            set: &descriptor_set,
            binding: 1,
            array_offset: 0,
            descriptors: Some(out_descriptor),
        };

        device.write_descriptor_sets(vec![in_descriptor_set_write, out_descriptor_set_write]);

        vec![descriptor_set]
    }

    unsafe fn create_command_buffer<'a>(
        buffer_size: u32,
        command_pool: &'a mut pool::CommandPool<back::Backend, Compute>,
        descriptor_sets: &[<back::Backend as Backend>::DescriptorSet],
        pipeline_layout: &'a <back::Backend as Backend>::PipelineLayout,
        pipeline: &<back::Backend as Backend>::ComputePipeline,
    ) -> command::CommandBuffer<back::Backend, Compute, command::OneShot, command::Primary>
    {
        let mut command_buffer: command::CommandBuffer<back::Backend, Compute, command::OneShot, command::Primary> = command_pool.acquire_command_buffer();

        command_buffer.begin();
        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_compute_descriptor_sets(pipeline_layout, 0, descriptor_sets, &[]);
        command_buffer.dispatch([buffer_size/mem::size_of::<i32>() as u32, 1, 1]);
        command_buffer.finish();

        command_buffer
    }

    unsafe fn execute_compute(&mut self) {
        let submission = queue::Submission {
            command_buffers: std::iter::once(&self.command_buffer),
            wait_semaphores: std::iter::empty(),
            signal_semaphores: std::iter::empty(),
        };

        let calculation_completed_fence = self.device.create_fence(false).unwrap();
        self.command_queues[0].submit(submission, Some(&calculation_completed_fence));
        self.device.wait_for_fence(&calculation_completed_fence, std::u64::MAX).unwrap();
        self.device.destroy_fence(calculation_completed_fence);
    }

    unsafe fn fill_payload(&mut self) {
        for j in 0isize..(self.buffer_size as isize) {
            ptr::copy(&rand::random::<u8>(), self.payload.offset(j), 1);
        }
    }

    unsafe fn check_result(&self) {
        for j in 0isize..(self.buffer_size as isize) {
            if ptr::read(self.payload.offset(j)) != ptr::read(self.result.offset(j)) {
                println!("Check failed: difference exists between payload and result.");
                return;
            }
        }
        println!("Check successful: payload is the same as result.");
    }

    unsafe fn clean_up(self) {
        let device = &self.device;

        device.destroy_command_pool(self.command_pool.into_raw());

        device.destroy_compute_pipeline(self.compute_pipeline);

        device.destroy_descriptor_set_layout(self.descriptor_set_layout);

        device.destroy_pipeline_layout(self.pipeline_layout);
    }
}
