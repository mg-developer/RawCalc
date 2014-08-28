#include "ProgGPU.h"
// includes, system
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
// includes CUDA Runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 

#include "Logging.h"

template<typename T>
void printPTR(const ManagedPtr<T> &p, int size) {
    for(int i=0; i != size; ++i)
        std::cout << p[i] << " ";
    std::cout << std::endl;
}


// This function returns the best GPU based on performance
std::pair<int, int> getMaxGflopsDeviceId() {
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;

    cuInit(0);
    checkCudaErrors(cuDeviceGetCount(&device_count));

    // Find the best major SM Architecture GPU device
    while (current_device < device_count) {
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major > 0 && major < 9999) {
            best_SM_arch = MAX(best_SM_arch, major);
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
    while (current_device < device_count) {
        checkCudaErrors(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, current_device));
        checkCudaErrors(cuDeviceGetAttribute(&clockRate,           CU_DEVICE_ATTRIBUTE_CLOCK_RATE,           current_device));
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major == 9999 && minor == 9999) {
            sm_per_multiproc = 1;
        }
        else {
            sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
        }

        int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;

        if (compute_perf  > max_compute_perf) {
            // If we find GPU with SM major > 2, search only these
            if (best_SM_arch > 2) {
                // If our device==dest_SM_arch, choose this, or else pass
                if (major == best_SM_arch) {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
            else {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }
        }

        ++current_device;
    }

    //Best device
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, max_perf_device);
    LOG() << "CUDA device [" << deviceProps.name << "]" << std::endl;

    return std::make_pair(max_perf_device, max_compute_perf);
}


__global__ void increment_kernel(int *g_data, int inc_value)
{
    const int i = inc_value;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + i;
}

__global__ void merge_kernel(int *static_tab, int *result)
{
    __shared__ float s_x[32];
    int t = threadIdx.x;
    for(int i = 0; i< 32; ++i)
        s_x[i] = static_tab[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = blockDim.x;
}

int calcOnGPU(const ProcessParams &p) {

    cudaDeviceProp cudaProps;
    cudaGetDeviceProperties(&cudaProps, 0);
    const int constant_mem = cudaProps.totalConstMem;
    const int shrared_mem = cudaProps.sharedMemPerBlock;
    const int threads_max = cudaProps.maxThreadsPerBlock;
    const int avail_mem = cudaProps.totalGlobalMem;

    const int inc_mem = 1024;

    LOG() << "Mb, Avail: " << avail_mem/1024.0/1024.0 << "Mb" << std::endl;

    printPTR(p.data,80);

    int *inc_data_dev;
    cudaMalloc((void **)&inc_data_dev, p.data.bytes());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    //Common stage
/*    int params[] = { p.recordCount, p.ballSet,
                     p.combinationCount, p.ballPerComb,
                     p.lengthCount, 0,
                     p.combinationIndexCount, p.combinationIndexCount/p.ballSet
                   };

    memcpy(PARAMS_soft, params, sizeof(int)*8);
    cudaMemcpyToSymbolAsync(PARAMS_dev, params, sizeof(int)*8);
*/

    //Stage I

    cudaMemcpyAsync(inc_data_dev, p.data(), p.data.bytes(), cudaMemcpyHostToDevice);
  //  cudaMemcpyAsync(comb_data_dev, p.combinationData(), comb_mem, cudaMemcpyHostToDevice);
  //  cudaMemsetAsync(length_data_dev, 0x0, length_mem);

    dim3 task1_threads;
    dim3 task1_blocks;
    if(p.count < threads_max) {
        task1_threads = dim3(p.count, 1);
        task1_blocks  = 1;
    } else {
        int n=p.count;
        task1_threads = dim3(threads_max, 1);
        task1_blocks  = dim3(n / task1_threads.x, 1);
    }

    //TODO: Process    
    increment_kernel <<<task1_blocks, task1_threads>>>(inc_data_dev, 123);

    cudaMemcpyAsync((void*)p.data(),  inc_data_dev, p.data.bytes(), cudaMemcpyDeviceToHost);

    cudaFree(inc_data_dev);

    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    printPTR(p.data,80);

    cudaEventElapsedTime(&gpu_time, start, stop);
    // print the cpu and gpu times
    LOG() << "time spent executing by the GPU: " <<  gpu_time << "ms,  Counts: " << counter << std::endl;

    cudaFree(inc_data_dev);
    return 0;
}


