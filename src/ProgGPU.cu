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
        LOG() << p[i] << " ";
    LOG() << std::endl;
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

int findBestGPU() {

    cudaSetDevice(0);
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    LOG() << "CUDA device [" << deviceProps.name << "]" << std::endl;
    LOG() << " Processors: " << _ConvertSMVer2Cores(deviceProps.major, deviceProps.minor) * deviceProps.multiProcessorCount << std::endl;
    LOG() << " Clock rate: " << (deviceProps.clockRate / 1000) << " MHz" << std::endl;
    LOG() << "     Memory: " << (deviceProps.totalGlobalMem / 1024)/1024 << " Mb" << std::endl;
    return 0;
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
    int n = p.data.size();
    dim3 task1_threads;
    dim3 task1_blocks;


    LOG() << "Thr-max: " << threads_max << " el:" << n << std::endl;
    if(n < threads_max) {
        task1_threads = dim3(n, 1);
        task1_blocks  = 1;
    } else {
        task1_threads = dim3(threads_max, 1);
        task1_blocks  = dim3(n / task1_threads.x, 1);
    }

    LOG() << "block:" <<  task1_blocks.x  << " thr:" << task1_threads.x << std::endl;
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




