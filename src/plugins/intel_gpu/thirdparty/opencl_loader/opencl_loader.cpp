// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <dlfcn.h>

namespace {

void* get_opencl_library() {
    static void* const library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    return library;
}

template <typename Function>
Function get_opencl_function(const char* name) {
    const auto library = get_opencl_library();
    return library == nullptr ? nullptr : reinterpret_cast<Function>(dlsym(library, name));
}

}  // namespace

#define OV_GPU_DEFINE_OPENCL_FUNCTION(return_type, function_name, parameters, arguments, on_failure) \
    extern "C" CL_API_ENTRY return_type CL_API_CALL function_name parameters {                       \
        using function_type = return_type(CL_API_CALL*) parameters;                                  \
        static const auto function = get_opencl_function<function_type>(#function_name);             \
        if (function == nullptr) {                                                                   \
            on_failure;                                                                              \
        }                                                                                            \
        return function arguments;                                                                   \
    }

#define OV_GPU_FORWARD_CL_INT(function_name, parameters, arguments) \
    OV_GPU_DEFINE_OPENCL_FUNCTION(cl_int, function_name, parameters, arguments, return CL_INVALID_OPERATION)

#define OV_GPU_FORWARD_CL_OBJECT(return_type, function_name, parameters, arguments) \
    OV_GPU_DEFINE_OPENCL_FUNCTION(                                                  \
        return_type,                                                                \
        function_name,                                                              \
        parameters,                                                                 \
        arguments,                                                                  \
        if (errcode_ret != nullptr) { *errcode_ret = CL_INVALID_OPERATION; } return nullptr)

OV_GPU_DEFINE_OPENCL_FUNCTION(
    cl_int,
    clGetPlatformIDs,
    (cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms),
    (num_entries, platforms, num_platforms),
    if (num_platforms != nullptr) { *num_platforms = 0; } return CL_PLATFORM_NOT_FOUND_KHR)

OV_GPU_FORWARD_CL_INT(clGetPlatformInfo,
                      (cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (platform, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_INT(clGetDeviceIDs,
                      (cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices),
                      (platform, device_type, num_entries, devices, num_devices))

OV_GPU_FORWARD_CL_INT(clGetDeviceInfo,
                      (cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (device, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_INT(
    clCreateSubDevices,
    (cl_device_id in_device, const cl_device_partition_property* properties, cl_uint num_devices, cl_device_id* out_devices, cl_uint* num_devices_ret),
    (in_device, properties, num_devices, out_devices, num_devices_ret))

OV_GPU_FORWARD_CL_INT(clRetainDevice, (cl_device_id device), (device))
OV_GPU_FORWARD_CL_INT(clReleaseDevice, (cl_device_id device), (device))

OV_GPU_FORWARD_CL_INT(clGetDeviceAndHostTimer,
                      (cl_device_id device, cl_ulong* device_timestamp, cl_ulong* host_timestamp),
                      (device, device_timestamp, host_timestamp))

OV_GPU_FORWARD_CL_OBJECT(cl_context,
                         clCreateContext,
                         (const cl_context_properties* properties,
                          cl_uint num_devices,
                          const cl_device_id* devices,
                          void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
                          void* user_data,
                          cl_int* errcode_ret),
                         (properties, num_devices, devices, pfn_notify, user_data, errcode_ret))

OV_GPU_FORWARD_CL_INT(clRetainContext, (cl_context context), (context))
OV_GPU_FORWARD_CL_INT(clReleaseContext, (cl_context context), (context))

OV_GPU_FORWARD_CL_INT(clGetContextInfo,
                      (cl_context context, cl_context_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (context, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_command_queue,
                         clCreateCommandQueueWithProperties,
                         (cl_context context, cl_device_id device, const cl_queue_properties* properties, cl_int* errcode_ret),
                         (context, device, properties, errcode_ret))

OV_GPU_FORWARD_CL_INT(clRetainCommandQueue, (cl_command_queue command_queue), (command_queue))
OV_GPU_FORWARD_CL_INT(clReleaseCommandQueue, (cl_command_queue command_queue), (command_queue))

OV_GPU_FORWARD_CL_INT(
    clGetCommandQueueInfo,
    (cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
    (command_queue, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_mem,
                         clCreateBuffer,
                         (cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret),
                         (context, flags, size, host_ptr, errcode_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_mem,
                         clCreateBufferWithProperties,
                         (cl_context context, const cl_mem_properties* properties, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret),
                         (context, properties, flags, size, host_ptr, errcode_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_mem,
                         clCreateSubBuffer,
                         (cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void* buffer_create_info, cl_int* errcode_ret),
                         (buffer, flags, buffer_create_type, buffer_create_info, errcode_ret))

OV_GPU_FORWARD_CL_OBJECT(
    cl_mem,
    clCreateImage,
    (cl_context context, cl_mem_flags flags, const cl_image_format* image_format, const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret),
    (context, flags, image_format, image_desc, host_ptr, errcode_ret))

OV_GPU_FORWARD_CL_INT(clRetainMemObject, (cl_mem memobj), (memobj))
OV_GPU_FORWARD_CL_INT(clReleaseMemObject, (cl_mem memobj), (memobj))

OV_GPU_FORWARD_CL_INT(clGetMemObjectInfo,
                      (cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (memobj, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_INT(clGetImageInfo,
                      (cl_mem image, cl_image_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (image, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_program,
                         clCreateProgramWithSource,
                         (cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret),
                         (context, count, strings, lengths, errcode_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_program,
                         clCreateProgramWithBinary,
                         (cl_context context,
                          cl_uint num_devices,
                          const cl_device_id* device_list,
                          const size_t* lengths,
                          const unsigned char** binaries,
                          cl_int* binary_status,
                          cl_int* errcode_ret),
                         (context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret))

OV_GPU_FORWARD_CL_INT(clRetainProgram, (cl_program program), (program))
OV_GPU_FORWARD_CL_INT(clReleaseProgram, (cl_program program), (program))

OV_GPU_FORWARD_CL_INT(clBuildProgram,
                      (cl_program program,
                       cl_uint num_devices,
                       const cl_device_id* device_list,
                       const char* options,
                       void(CL_CALLBACK* pfn_notify)(cl_program, void*),
                       void* user_data),
                      (program, num_devices, device_list, options, pfn_notify, user_data))

OV_GPU_FORWARD_CL_INT(clGetProgramInfo,
                      (cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (program, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_INT(
    clGetProgramBuildInfo,
    (cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
    (program, device, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_kernel, clCreateKernel, (cl_program program, const char* kernel_name, cl_int* errcode_ret), (program, kernel_name, errcode_ret))

OV_GPU_FORWARD_CL_INT(clCreateKernelsInProgram,
                      (cl_program program, cl_uint num_kernels, cl_kernel* kernels, cl_uint* num_kernels_ret),
                      (program, num_kernels, kernels, num_kernels_ret))

OV_GPU_FORWARD_CL_INT(clRetainKernel, (cl_kernel kernel), (kernel))
OV_GPU_FORWARD_CL_INT(clReleaseKernel, (cl_kernel kernel), (kernel))

OV_GPU_FORWARD_CL_INT(clSetKernelArg, (cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value), (kernel, arg_index, arg_size, arg_value))

OV_GPU_FORWARD_CL_INT(clGetKernelInfo,
                      (cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (kernel, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_INT(clWaitForEvents, (cl_uint num_events, const cl_event* event_list), (num_events, event_list))

OV_GPU_FORWARD_CL_INT(clGetEventInfo,
                      (cl_event event, cl_event_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (event, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_OBJECT(cl_event, clCreateUserEvent, (cl_context context, cl_int* errcode_ret), (context, errcode_ret))

OV_GPU_FORWARD_CL_INT(clRetainEvent, (cl_event event), (event))
OV_GPU_FORWARD_CL_INT(clReleaseEvent, (cl_event event), (event))
OV_GPU_FORWARD_CL_INT(clSetUserEventStatus, (cl_event event, cl_int execution_status), (event, execution_status))

OV_GPU_FORWARD_CL_INT(clSetEventCallback,
                      (cl_event event, cl_int command_exec_callback_type, void(CL_CALLBACK* pfn_notify)(cl_event, cl_int, void*), void* user_data),
                      (event, command_exec_callback_type, pfn_notify, user_data))

OV_GPU_FORWARD_CL_INT(clGetEventProfilingInfo,
                      (cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret),
                      (event, param_name, param_value_size, param_value, param_value_size_ret))

OV_GPU_FORWARD_CL_INT(clFlush, (cl_command_queue command_queue), (command_queue))
OV_GPU_FORWARD_CL_INT(clFinish, (cl_command_queue command_queue), (command_queue))

OV_GPU_FORWARD_CL_INT(clEnqueueReadBuffer,
                      (cl_command_queue command_queue,
                       cl_mem buffer,
                       cl_bool blocking_read,
                       size_t offset,
                       size_t size,
                       void* ptr,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueWriteBuffer,
                      (cl_command_queue command_queue,
                       cl_mem buffer,
                       cl_bool blocking_write,
                       size_t offset,
                       size_t size,
                       const void* ptr,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueFillBuffer,
                      (cl_command_queue command_queue,
                       cl_mem buffer,
                       const void* pattern,
                       size_t pattern_size,
                       size_t offset,
                       size_t size,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueCopyBuffer,
                      (cl_command_queue command_queue,
                       cl_mem src_buffer,
                       cl_mem dst_buffer,
                       size_t src_offset,
                       size_t dst_offset,
                       size_t size,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueReadImage,
                      (cl_command_queue command_queue,
                       cl_mem image,
                       cl_bool blocking_read,
                       const size_t* origin,
                       const size_t* region,
                       size_t row_pitch,
                       size_t slice_pitch,
                       void* ptr,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(
    clEnqueueWriteImage,
    (cl_command_queue command_queue,
     cl_mem image,
     cl_bool blocking_write,
     const size_t* origin,
     const size_t* region,
     size_t input_row_pitch,
     size_t input_slice_pitch,
     const void* ptr,
     cl_uint num_events_in_wait_list,
     const cl_event* event_wait_list,
     cl_event* event),
    (command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueFillImage,
                      (cl_command_queue command_queue,
                       cl_mem image,
                       const void* fill_color,
                       const size_t* origin,
                       const size_t* region,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, image, fill_color, origin, region, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueCopyImage,
                      (cl_command_queue command_queue,
                       cl_mem src_image,
                       cl_mem dst_image,
                       const size_t* src_origin,
                       const size_t* dst_origin,
                       const size_t* region,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list, event))

OV_GPU_DEFINE_OPENCL_FUNCTION(
    void*,
    clEnqueueMapBuffer,
    (cl_command_queue command_queue,
     cl_mem buffer,
     cl_bool blocking_map,
     cl_map_flags map_flags,
     size_t offset,
     size_t size,
     cl_uint num_events_in_wait_list,
     const cl_event* event_wait_list,
     cl_event* event,
     cl_int* errcode_ret),
    (command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret),
    if (errcode_ret != nullptr) { *errcode_ret = CL_INVALID_OPERATION; } return nullptr)

OV_GPU_DEFINE_OPENCL_FUNCTION(
    void*,
    clEnqueueMapImage,
    (cl_command_queue command_queue,
     cl_mem image,
     cl_bool blocking_map,
     cl_map_flags map_flags,
     const size_t* origin,
     const size_t* region,
     size_t* image_row_pitch,
     size_t* image_slice_pitch,
     cl_uint num_events_in_wait_list,
     const cl_event* event_wait_list,
     cl_event* event,
     cl_int* errcode_ret),
    (command_queue,
     image,
     blocking_map,
     map_flags,
     origin,
     region,
     image_row_pitch,
     image_slice_pitch,
     num_events_in_wait_list,
     event_wait_list,
     event,
     errcode_ret),
    if (errcode_ret != nullptr) { *errcode_ret = CL_INVALID_OPERATION; } return nullptr)

OV_GPU_FORWARD_CL_INT(
    clEnqueueUnmapMemObject,
    (cl_command_queue command_queue, cl_mem memobj, void* mapped_ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event),
    (command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueNDRangeKernel,
                      (cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t* global_work_offset,
                       const size_t* global_work_size,
                       const size_t* local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* event),
                      (command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueMarkerWithWaitList,
                      (cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event),
                      (command_queue, num_events_in_wait_list, event_wait_list, event))

OV_GPU_FORWARD_CL_INT(clEnqueueBarrierWithWaitList,
                      (cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event),
                      (command_queue, num_events_in_wait_list, event_wait_list, event))

OV_GPU_DEFINE_OPENCL_FUNCTION(void*,
                              clGetExtensionFunctionAddressForPlatform,
                              (cl_platform_id platform, const char* func_name),
                              (platform, func_name),
                              return nullptr)

#undef OV_GPU_FORWARD_CL_OBJECT
#undef OV_GPU_FORWARD_CL_INT
#undef OV_GPU_DEFINE_OPENCL_FUNCTION
