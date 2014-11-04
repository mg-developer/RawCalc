#include "Logging.h"
#include "RawStructs.h"
#include "ManagedMem.h"
#include <memory>

#include <helper_cuda.h>

static MDB destinationData[6];

__global__ void proc_kernel_conv_14_16_max(int last_chunk, int bl, float maximize, unsigned char *in, unsigned char *out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = idx * 14;
    int tt = idx * 16;
    unsigned int senselA, senselB, senselC, senselD, senselE, senselF, senselG, senselH;
    unsigned int maximizer = maximize;

    if(idx < last_chunk) {

        unsigned int in_0 = (unsigned int)in[t];
        unsigned int in_1 = (unsigned int)in[t+1];
        unsigned int in_2 = (unsigned int)in[t+2];
        unsigned int in_3 = (unsigned int)in[t+3];
        unsigned int in_4 = (unsigned int)in[t+4];
        unsigned int in_5 = (unsigned int)in[t+5];
        unsigned int in_6 = (unsigned int)in[t+6];
        unsigned int in_7 = (unsigned int)in[t+7];
        unsigned int in_8 = (unsigned int)in[t+8];
        unsigned int in_9 = (unsigned int)in[t+9];
        unsigned int in_10 = (unsigned int)in[t+10];
        unsigned int in_11 = (unsigned int)in[t+11];
        unsigned int in_12 = (unsigned int)in[t+12];
        unsigned int in_13 = (unsigned int)in[t+13];
        unsigned int in_14 = (unsigned int)in[t+14];
        unsigned int in_15 = (unsigned int)in[t+15];


        senselA = ((in_0 >> 2)              | (in_1 << 6));
        senselB = ((in_0 & 0x3) << 12)      | (in_3 << 4)       | (in_2 >> 4);
        senselC = ((in_2 & 0x0f) << 10)     | (in_5 << 2)       | (in_4 >> 6);
        senselD = ((in_4 & 0x3f) << 8)      | (in_7);
        senselE = ((in_9 >> 2)              | (in_6 << 6));
        senselF = ((in_9 & 0x3) << 12)      | (in_8 << 4)       | (in_11 >> 4);
        senselG = ((in_11 & 0x0f) << 10)    | (in_10 << 2)      | (in_13 >> 6);
        senselH = ((in_13 & 0x3f) << 8)     | (in_12);

// debias sensel
        senselA -= bl;
        senselB -= bl;
        senselC -= bl;
        senselD -= bl;
        senselE -= bl;
        senselF -= bl;
        senselG -= bl;
        senselH -= bl;


// maximize to 16bit
        senselA = (senselA * maximizer);
        senselB = (senselB * maximizer);
        senselC = (senselC * maximizer);
        senselD = (senselD * maximizer);
        senselE = (senselE * maximizer);
        senselF = (senselF * maximizer);
        senselG = (senselG * maximizer);
        senselH = (senselH * maximizer);

// do max on overflow
        if (senselA > 65535) senselA = 65535;
        if (senselB > 65535) senselB = 65535;
        if (senselC > 65535) senselC = 65535;
        if (senselD > 65535) senselD = 65535;
        if (senselE > 65535) senselE = 65535;
        if (senselF > 65535) senselF = 65535;
        if (senselG > 65535) senselG = 65535;
        if (senselH > 65535) senselH = 65535;

// -- react on underflow
        if (senselA < 0) senselA = 0;
        if (senselB < 0) senselB = 0;
        if (senselC < 0) senselC = 0;
        if (senselD < 0) senselD = 0;
        if (senselE < 0) senselE = 0;
        if (senselF < 0) senselF = 0;
        if (senselG < 0) senselG = 0;
        if (senselH < 0) senselH = 0;

        out[tt] = (unsigned char)(senselA & 0xff);
        out[tt+1] = (unsigned char)(senselA >> 8);

        out[tt+2] = (unsigned char)(senselB & 0xff);
        out[tt+3] = (unsigned char)(senselB >> 8);

        out[tt+4] = (unsigned char)(senselC & 0xff);
        out[tt+5] = (unsigned char)(senselC >> 8);

        out[tt+6] = (unsigned char)(senselD & 0xff);
        out[tt+7] = (unsigned char)(senselD >> 8);

        out[tt+8] = (unsigned char)(senselE & 0xff);
        out[tt+9] = (unsigned char)(senselE >> 8);

        out[tt+10] = (unsigned char)(senselF & 0xff);
        out[tt+11] = (unsigned char)(senselF >> 8);

        out[tt+12] = (unsigned char)(senselG & 0xff);
        out[tt+13] = (unsigned char)(senselG >> 8);

        out[tt+14] = (unsigned char)(senselH & 0xff);
        out[tt+15] = (unsigned char)(senselH >> 8);

    }

}


__global__ void proc_kernel_conv_14_16(int last_chunk, unsigned char *in, unsigned char *out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = idx * 14;
    int tt = idx * 16;
    int senselA, senselB, senselC, senselD, senselE, senselF, senselG, senselH;

    if(idx < last_chunk) {

        unsigned int in_0 = (unsigned int)in[t];
        unsigned int in_1 = (unsigned int)in[t+1];
        unsigned int in_2 = (unsigned int)in[t+2];
        unsigned int in_3 = (unsigned int)in[t+3];
        unsigned int in_4 = (unsigned int)in[t+4];
        unsigned int in_5 = (unsigned int)in[t+5];
        unsigned int in_6 = (unsigned int)in[t+6];
        unsigned int in_7 = (unsigned int)in[t+7];
        unsigned int in_8 = (unsigned int)in[t+8];
        unsigned int in_9 = (unsigned int)in[t+9];
        unsigned int in_10 = (unsigned int)in[t+10];
        unsigned int in_11 = (unsigned int)in[t+11];
        unsigned int in_12 = (unsigned int)in[t+12];
        unsigned int in_13 = (unsigned int)in[t+13];
        unsigned int in_14 = (unsigned int)in[t+14];
        unsigned int in_15 = (unsigned int)in[t+15];


        senselA = ((in_0 >> 2)              | (in_1 << 6));
        senselB = ((in_0 & 0x3) << 12)      | (in_3 << 4)       | (in_2 >> 4);
        senselC = ((in_2 & 0x0f) << 10)     | (in_5 << 2)       | (in_4 >> 6);
        senselD = ((in_4 & 0x3f) << 8)      | (in_7);
        senselE = ((in_9 >> 2)              | (in_6 << 6));
        senselF = ((in_9 & 0x3) << 12)      | (in_8 << 4)       | (in_11 >> 4);
        senselG = ((in_11 & 0x0f) << 10)    | (in_10 << 2)      | (in_13 >> 6);
        senselH = ((in_13 & 0x3f) << 8)     | (in_12);

        out[tt] = (unsigned char)(senselA & 0xff);
        out[tt+1] = (unsigned char)(senselA >> 8);

        out[tt+2] = (unsigned char)(senselB & 0xff);
        out[tt+3] = (unsigned char)(senselB >> 8);

        out[tt+4] = (unsigned char)(senselC & 0xff);
        out[tt+5] = (unsigned char)(senselC >> 8);

        out[tt+6] = (unsigned char)(senselD & 0xff);
        out[tt+7] = (unsigned char)(senselD >> 8);

        out[tt+8] = (unsigned char)(senselE & 0xff);
        out[tt+9] = (unsigned char)(senselE >> 8);

        out[tt+10] = (unsigned char)(senselF & 0xff);
        out[tt+11] = (unsigned char)(senselF >> 8);

        out[tt+12] = (unsigned char)(senselG & 0xff);
        out[tt+13] = (unsigned char)(senselG >> 8);

        out[tt+14] = (unsigned char)(senselH & 0xff);
        out[tt+15] = (unsigned char)(senselH >> 8);
    }

}

MDB &to16H(const metadata &metaData, const MDB &in) {

    cudaDeviceProp cudaProps;
    cudaGetDeviceProperties(&cudaProps, 0);
    const int constant_mem = cudaProps.totalConstMem;
    const int shrared_mem = cudaProps.sharedMemPerBlock;
    const int threads_max = cudaProps.maxThreadsPerBlock;
    const int blocks_max[3] = { cudaProps.maxGridSize[0], cudaProps.maxGridSize[1],  cudaProps.maxGridSize[2]}; 
    const int mp_max = cudaProps.multiProcessorCount;
    const int avail_mem = cudaProps.totalGlobalMem;

    const int resx = metaData.xResolution;
    const int resy = metaData.yResolution;
    const int bl = metaData.blackLevelOld;
    const bool maximize = metaData.maximize;
    const double maximizer = metaData.maximizer;
    unsigned int pixels = resx * resy;
    unsigned int chunks = pixels * 14 / 8;
    const unsigned char* source = in();

    destinationData[0].reset(std::size_t(pixels * 16 / 8));
    MDB &dst = destinationData[0];

    cudaDeviceSynchronize();

    //CUDA timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //CUDA memory copy
    unsigned char *input_data_dev;
    unsigned char *output_data_dev;

    checkCudaErrors(cudaMalloc((void **)&input_data_dev, in.bytes()));
    checkCudaErrors(cudaMalloc((void **)&output_data_dev, dst.bytes()));
//    checkCudaErrors(cudaMemset(output_data_dev, 0, dst.bytes()));
    checkCudaErrors(cudaMemcpyAsync(input_data_dev, in(), in.bytes(), cudaMemcpyHostToDevice/*, stream*/));

    //CUDA plan
    int pixels_per_thread = 8;
    int data_to_process = pixels / pixels_per_thread;

    //CUDA dim set
    int block_count = data_to_process / threads_max;
    int block_count_rest = data_to_process % threads_max;
    if(block_count_rest)
        block_count++;

    LOG() << "Plan:" <<  " chk:" << chunks << " tile:" << threads_max << 
            " block:" << block_count << "/" << block_count_rest << std::endl;

    dim3 task_blocks(block_count,1);
    dim3 task_threads(threads_max, 1);

    //CUDA kernel
    if(maximize) {
        proc_kernel_conv_14_16_max <<< task_blocks, task_threads /*, 0, stream*/>>>(chunks, bl, maximizer, input_data_dev, output_data_dev);
    }
    else {
        proc_kernel_conv_14_16 <<< task_blocks, task_threads, 0 /*, stream*/>>>(chunks, input_data_dev, output_data_dev);
    }

    //CUDA finish
    checkCudaErrors(cudaMemcpyAsync(dst(),  output_data_dev, dst.bytes(), cudaMemcpyDeviceToHost/*, stream*/));

    cudaEventRecord(stop, 0);

    int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
      counter++;
    }

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    LOG() << "Finished.\nTime spent executing by the GPU: " <<  gpu_time << "ms,  Counts: " << counter << std::endl;


/*    for(int i=0; i<40; ++i)
        LOG() << std::hex << (int)dst[i] << " ";
    LOG() << std::endl;
*/
    cudaFree(input_data_dev);
    cudaFree(output_data_dev);


    return destinationData[0];
}

MDB &to16(const metadata &metaData, const MDB &in)
{
    const int resx = metaData.xResolution;
    const int resy = metaData.yResolution;
    const int bl = metaData.blackLevelOld;
    bool maximize = metaData.maximize;
    const double maximizer = metaData.maximizer;
    unsigned int chunks = resx * resy * 14 / 8;
    const unsigned char* source = in();

    maximize = false;

    destinationData[0].reset(std::size_t(chunks / 14 * 16));
    MDB &Dest = destinationData[0];


    unsigned int tt = 0;
    int senselA, senselB, senselC, senselD, senselE, senselF, senselG, senselH;
    for (unsigned int t = 0; t < chunks; t += 14)
    {

        unsigned int in_0 = (unsigned int)source[t];
        unsigned int in_1 = (unsigned int)source[t+1];
        unsigned int in_2 = (unsigned int)source[t+2];
        unsigned int in_3 = (unsigned int)source[t+3];
        unsigned int in_4 = (unsigned int)source[t+4];
        unsigned int in_5 = (unsigned int)source[t+5];
        unsigned int in_6 = (unsigned int)source[t+6];
        unsigned int in_7 = (unsigned int)source[t+7];
        unsigned int in_8 = (unsigned int)source[t+8];
        unsigned int in_9 = (unsigned int)source[t+9];
        unsigned int in_10 = (unsigned int)source[t+10];
        unsigned int in_11 = (unsigned int)source[t+11];
        unsigned int in_12 = (unsigned int)source[t+12];
        unsigned int in_13 = (unsigned int)source[t+13];
        unsigned int in_14 = (unsigned int)source[t+14];
        unsigned int in_15 = (unsigned int)source[t+15];


        if (maximize == true)
        {
            senselA = ( (in_0 >> 2) | ( in_1 << 6));
            senselB = ( (in_0 & 0x3) << 12) | (in_3 << 4) | (in_2 >> 4);
            senselC = ((in_2 & 0x0f) << 10) | (in_5 << 2) | (in_4 >> 6);
            senselD = ((in_4 & 0x3f) << 8) | (in_7);
            senselE = (in_9 >> 2) | (in_6 << 6);
            senselF = ((in_9 & 0x3) << 12) | (in_8 << 4) | (in_11 >> 4);
            senselG = ((in_11 & 0x0f) << 10) | (in_10 << 2) | (in_13 >> 6);
            senselH = ((in_13 & 0x3f) << 8) | in_12;

            // debias sensel
            senselA -= bl;
            senselB -= bl;
            senselC -= bl;
            senselD -= bl;
            senselE -= bl;
            senselF -= bl;
            senselG -= bl;
            senselH -= bl;

            // maximize to 16bit
            senselA = (int)(senselA * maximizer);
            senselB = (int)(senselB * maximizer);
            senselC = (int)(senselC * maximizer);
            senselD = (int)(senselD * maximizer);
            senselE = (int)(senselE * maximizer);
            senselF = (int)(senselF * maximizer);
            senselG = (int)(senselG * maximizer);
            senselH = (int)(senselH * maximizer);

            // do max on overflow
            if (senselA > 65535) senselA = 65535;
            if (senselB > 65535) senselB = 65535;
            if (senselC > 65535) senselC = 65535;
            if (senselD > 65535) senselD = 65535;
            if (senselE > 65535) senselE = 65535;
            if (senselF > 65535) senselF = 65535;
            if (senselG > 65535) senselG = 65535;
            if (senselH > 65535) senselH = 65535;

            // -- react on underflow
            if (senselA < 0) senselA = 0;
            if (senselB < 0) senselB = 0;
            if (senselC < 0) senselC = 0;
            if (senselD < 0) senselD = 0;
            if (senselE < 0) senselE = 0;
            if (senselF < 0) senselF = 0;
            if (senselG < 0) senselG = 0;
            if (senselH < 0) senselH = 0;

        }
        else
        {
            senselA = ( (in_0 >> 2) | ( in_1 << 6));
            senselB = ( (in_0 & 0x3) << 12) | (in_3 << 4) | (in_2 >> 4);
            senselC = ((in_2 & 0x0f) << 10) | (in_5 << 2) | (in_4 >> 6);
            senselD = ((in_4 & 0x3f) << 8) | (in_7);
            senselE = (in_9 >> 2) | (in_6 << 6);
            senselF = ((in_9 & 0x3) << 12) | (in_8 << 4) | (in_11 >> 4);
            senselG = ((in_11 & 0x0f) << 10) | (in_10 << 2) | (in_13 >> 6);
            senselH = ((in_13 & 0x3f) << 8) | in_12;
        }

        Dest[tt++] = (unsigned char)(senselA & 0xff);
        Dest[tt++] = (unsigned char)(senselA >> 8);

        Dest[tt++] = (unsigned char)(senselB & 0xff);
        Dest[tt++] = (unsigned char)(senselB >> 8);

        Dest[tt++] = (unsigned char)(senselC & 0xff);
        Dest[tt++] = (unsigned char)(senselC >> 8);

        Dest[tt++] = (unsigned char)(senselD & 0xff);
        Dest[tt++] = (unsigned char)(senselD >> 8);

        Dest[tt++] = (unsigned char)(senselE & 0xff);
        Dest[tt++] = (unsigned char)(senselE >> 8);

        Dest[tt++] = (unsigned char)(senselF & 0xff);
        Dest[tt++] = (unsigned char)(senselF >> 8);

        Dest[tt++] = (unsigned char)(senselG & 0xff);
        Dest[tt++] = (unsigned char)(senselG >> 8);

        Dest[tt++] = (unsigned char)(senselH & 0xff);
        Dest[tt++] = (unsigned char)(senselH >> 8);

    }

    return destinationData[0];
}

// 5DIII valuerange
// 14bit original - 2048-15.000 = ~12.900
// 16bit - 8192-60.000 - maximized 0-65535
// 12bit - 512-3750 = ~3.200 - maximized 0-4095 (

MDB &from16to12(const metadata &metaData, const MDB &in)
{
    // preparing variables
    int resx = metaData.xResolution;
    int resy = metaData.yResolution;
    const unsigned char* source = in();
    // ------------- and go ----

    unsigned int chunks = resx * resy * 16 / 8;
    destinationData[0].reset(std::size_t(chunks / 16 * 12 + 72));
    MDB &Dest = destinationData[0];

    unsigned int tt = 0;
    int senselA, senselB, senselC, senselD, senselE, senselF, senselG, senselH;
    int senselI, senselJ, senselK, senselL, senselM, senselN, senselO, senselP;
    int senselQ, senselR, senselS, senselT, senselU, senselV, senselW, senselX;

    for (unsigned int t = 0; t < chunks; t += 48)
    {
        //read 16bit data and shift 4 bits.
        senselA = (source[t] | (source[t + 1] << 8)) >> 4;
        senselB = (source[t + 2] | (source[t + 3] << 8)) >> 4;
        senselC = (source[t + 4] | (source[t + 5] << 8)) >> 4;
        senselD = (source[t + 6] | (source[t + 7] << 8)) >> 4;
        senselE = (source[t + 8] | (source[t + 9] << 8)) >> 4;
        senselF = (source[t + 10] | (source[t + 11] << 8)) >> 4;
        senselG = (source[t + 12] | (source[t + 13] << 8)) >> 4;
        senselH = (source[t + 14] | (source[t + 15] << 8)) >> 4;

        senselI = (source[t + 16] | (source[t + 17] << 8)) >> 4;
        senselJ = (source[t + 18] | (source[t + 19] << 8)) >> 4;
        senselK = (source[t + 20] | (source[t + 21] << 8)) >> 4;
        senselL = (source[t + 22] | (source[t + 23] << 8)) >> 4;
        senselM = (source[t + 24] | (source[t + 25] << 8)) >> 4;
        senselN = (source[t + 26] | (source[t + 27] << 8)) >> 4;
        senselO = (source[t + 28] | (source[t + 29] << 8)) >> 4;
        senselP = (source[t + 30] | (source[t + 31] << 8)) >> 4;

        senselQ = (source[t + 32] | (source[t + 33] << 8)) >> 4;
        senselR = (source[t + 34] | (source[t + 35] << 8)) >> 4;
        senselS = (source[t + 36] | (source[t + 37] << 8)) >> 4;
        senselT = (source[t + 38] | (source[t + 39] << 8)) >> 4;
        senselU = (source[t + 40] | (source[t + 41] << 8)) >> 4;
        senselV = (source[t + 42] | (source[t + 43] << 8)) >> 4;
        senselW = (source[t + 44] | (source[t + 45] << 8)) >> 4;
        senselX = (source[t + 46] | (source[t + 47] << 8)) >> 4;

        Dest[tt++] = (unsigned char)((senselA >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselA & 0xF) << 4) | (senselB >> 8));
        Dest[tt++] = (unsigned char)(senselB & 0xff);

        Dest[tt++] = (unsigned char)((senselC >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselC & 0xF) << 4) | (senselD >> 8));
        Dest[tt++] = (unsigned char)(senselD & 0xff);

        Dest[tt++] = (unsigned char)((senselE >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselE & 0xF) << 4) | (senselF >> 8));
        Dest[tt++] = (unsigned char)(senselF & 0xff);
        //9
        Dest[tt++] = (unsigned char)((senselG >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselG & 0xF) << 4) | (senselH >> 8));
        Dest[tt++] = (unsigned char)(senselH & 0xff);

        Dest[tt++] = (unsigned char)((senselI >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselI & 0xF) << 4) | (senselJ >> 8));
        Dest[tt++] = (unsigned char)(senselJ & 0xff);

        Dest[tt++] = (unsigned char)((senselK >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselK & 0xF) << 4) | (senselL >> 8));
        Dest[tt++] = (unsigned char)(senselL & 0xff);
        //18
        Dest[tt++] = (unsigned char)((senselM >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselM & 0xF) << 4) | (senselN >> 8));
        Dest[tt++] = (unsigned char)(senselN & 0xff);

        Dest[tt++] = (unsigned char)((senselO >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselO & 0xF) << 4) | (senselP >> 8));
        Dest[tt++] = (unsigned char)(senselP & 0xff);

        Dest[tt++] = (unsigned char)((senselQ >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselQ & 0xF) << 4) | (senselR >> 8));
        Dest[tt++] = (unsigned char)(senselR & 0xff);
        //27
        Dest[tt++] = (unsigned char)((senselS >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselS & 0xF) << 4) | (senselT >> 8));
        Dest[tt++] = (unsigned char)(senselT & 0xff);

        Dest[tt++] = (unsigned char)((senselU >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselU & 0xF) << 4) | (senselV >> 8));
        Dest[tt++] = (unsigned char)(senselV & 0xff);

        Dest[tt++] = (unsigned char)((senselW >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselW & 0xF) << 4) | (senselX >> 8));
        Dest[tt++] = (unsigned char)(senselX & 0xff);
        //36
    }
    return destinationData[0];
}

MDB &to12(const metadata &metaData, const MDB &in)
{
    // preparing variables
    int resx = metaData.xResolution;
    int resy = metaData.yResolution;
    int bl = metaData.blackLevelOld;
    bool maximize = metaData.maximize;
    double maximizer = metaData.maximizer;
    const unsigned char* source = in();
    // ------------- and go ----

    unsigned int chunks = resx * resy * 14 / 8;
    destinationData[0].reset(std::size_t(chunks / 14 * 12 + 42));
    MDB &Dest = destinationData[0];


    unsigned int tt = 0;
    int senselA, senselB, senselC, senselD, senselE, senselF, senselG, senselH;
    int senselI, senselJ, senselK, senselL, senselM, senselN, senselO, senselP;
    int senselQ, senselR, senselS, senselT, senselU, senselV, senselW, senselX;

    for (unsigned int t = 0; t < chunks; t += 42)
    {
        if (maximize == true)
        {
            senselA = (int)((source[t] >> 2) | (source[t + 1] << 6)) - (int)bl;
            senselB = (int)(((source[t] & 0x3) << 12) | (source[t + 3] << 4) | (source[t + 2] >> 4)) - (int)bl;
            senselC = (int)(((source[t + 2] & 0x0f) << 10) | (source[t + 5] << 2) | (source[t + 4] >> 6)) - (int)bl;
            senselD = (int)(((source[t + 4] & 0x3f) << 8) | (source[t + 7])) - (int)bl;
            senselE = (int)((source[t + 9] >> 2) | (source[t + 6] << 6)) - (int)bl;
            senselF = (int)(((source[t + 9] & 0x3) << 12) | (source[t + 8] << 4) | (source[t + 11] >> 4)) - (int)bl;
            senselG = (int)(((source[t + 11] & 0x0f) << 10) | (source[t + 10] << 2) | (source[t + 13] >> 6)) - (int)bl;
            senselH = (int)(((source[t + 13] & 0x3f) << 8) | (source[t + 12])) - (int)bl;

            senselI = (int)((source[t + 14] >> 2) | (source[t + 15] << 6)) - (int)bl;
            senselJ = (int)(((source[t + 14] & 0x3) << 12) | (source[t + 17] << 4) | (source[t + 16] >> 4)) - (int)bl;
            senselK = (int)(((source[t + 16] & 0x0f) << 10) | (source[t + 19] << 2) | (source[t + 18] >> 6)) - (int)bl;
            senselL = (int)(((source[t + 18] & 0x3f) << 8) | (source[t + 21])) - (int)bl;
            senselM = (int)((source[t + 23] >> 2) | (source[t + 20] << 6)) - (int)bl;
            senselN = (int)(((source[t + 23] & 0x3) << 12) | (source[t + 22] << 4) | (source[t + 25] >> 4)) - (int)bl;
            senselO = (int)(((source[t + 25] & 0x0f) << 10) | (source[t + 24] << 2) | (source[t + 27] >> 6)) - (int)bl;
            senselP = (int)(((source[t + 27] & 0x3f) << 8) | (source[t + 26])) - (int)bl;

            senselQ = (int)((source[t + 28] >> 2) | (source[t + 29] << 6)) - (int)bl;
            senselR = (int)(((source[t + 28] & 0x3) << 12) | (source[t + 31] << 4) | (source[t + 30] >> 4)) - (int)bl;
            senselS = (int)(((source[t + 30] & 0x0f) << 10) | (source[t + 33] << 2) | (source[t + 32] >> 6)) - (int)bl;
            senselT = (int)(((source[t + 32] & 0x3f) << 8) | (source[t + 35])) - (int)bl;
            senselU = (int)((source[t + 37] >> 2) | (source[t + 34] << 6)) - (int)bl;
            senselV = (int)(((source[t + 37] & 0x3) << 12) | (source[t + 36] << 4) | (source[t + 39] >> 4)) - (int)bl;
            senselW = (int)(((source[t + 39] & 0x0f) << 10) | (source[t + 38] << 2) | (source[t + 41] >> 6)) - (int)bl;
            senselX = (int)(((source[t + 41] & 0x3f) << 8) | (source[t + 40])) - (int)bl;

            // maximize to 12bit
            senselA = (int)(senselA * maximizer);
            senselB = (int)(senselB * maximizer);
            senselC = (int)(senselC * maximizer);
            senselD = (int)(senselD * maximizer);
            senselE = (int)(senselE * maximizer);
            senselF = (int)(senselF * maximizer);
            senselG = (int)(senselG * maximizer);
            senselH = (int)(senselH * maximizer);
            senselI = (int)(senselI * maximizer);
            senselJ = (int)(senselJ * maximizer);
            senselK = (int)(senselK * maximizer);
            senselL = (int)(senselL * maximizer);
            senselM = (int)(senselM * maximizer);
            senselN = (int)(senselN * maximizer);
            senselO = (int)(senselO * maximizer);
            senselP = (int)(senselP * maximizer);
            senselQ = (int)(senselQ * maximizer);
            senselR = (int)(senselR * maximizer);
            senselS = (int)(senselS * maximizer);
            senselT = (int)(senselT * maximizer);
            senselU = (int)(senselU * maximizer);
            senselV = (int)(senselV * maximizer);
            senselW = (int)(senselW * maximizer);
            senselX = (int)(senselX * maximizer);

            // check on overflow
            if (senselA > 4095) senselA = 4095;
            if (senselB > 4095) senselB = 4095;
            if (senselC > 4095) senselC = 4095;
            if (senselD > 4095) senselD = 4095;
            if (senselE > 4095) senselE = 4095;
            if (senselF > 4095) senselF = 4095;
            if (senselG > 4095) senselG = 4095;
            if (senselH > 4095) senselH = 4095;
            if (senselI > 4095) senselI = 4095;
            if (senselJ > 4095) senselJ = 4095;
            if (senselK > 4095) senselK = 4095;
            if (senselL > 4095) senselL = 4095;
            if (senselM > 4095) senselM = 4095;
            if (senselN > 4095) senselN = 4095;
            if (senselO > 4095) senselO = 4095;
            if (senselP > 4095) senselP = 4095;
            if (senselQ > 4095) senselQ = 4095;
            if (senselR > 4095) senselR = 4095;
            if (senselS > 4095) senselS = 4095;
            if (senselT > 4095) senselT = 4095;
            if (senselU > 4095) senselU = 4095;
            if (senselV > 4095) senselV = 4095;
            if (senselW > 4095) senselW = 4095;
            if (senselX > 4095) senselX = 4095;


            // -- react on underflow
            if (senselA < 0) senselA = 0;
            if (senselB < 0) senselB = 0;
            if (senselC < 0) senselC = 0;
            if (senselD < 0) senselD = 0;
            if (senselE < 0) senselE = 0;
            if (senselF < 0) senselF = 0;
            if (senselG < 0) senselG = 0;
            if (senselH < 0) senselH = 0;
            if (senselI < 0) senselI = 0;
            if (senselJ < 0) senselJ = 0;
            if (senselK < 0) senselK = 0;
            if (senselL < 0) senselL = 0;
            if (senselM < 0) senselM = 0;
            if (senselN < 0) senselN = 0;
            if (senselO < 0) senselO = 0;
            if (senselP < 0) senselP = 0;
            if (senselQ < 0) senselQ = 0;
            if (senselR < 0) senselR = 0;
            if (senselS < 0) senselS = 0;
            if (senselT < 0) senselT = 0;
            if (senselU < 0) senselU = 0;
            if (senselV < 0) senselV = 0;
            if (senselW < 0) senselW = 0;
            if (senselX < 0) senselX = 0;

        }
        else
        {
            senselA = (int)((source[t] >> 2) | (source[t + 1] << 6));
            senselB = (int)(((source[t] & 0x3) << 12) | (source[t + 3] << 4) | (source[t + 2] >> 4));
            senselC = (int)(((source[t + 2] & 0x0f) << 10) | (source[t + 5] << 2) | (source[t + 4] >> 6));
            senselD = (int)(((source[t + 4] & 0x3f) << 8) | (source[t + 7]));
            senselE = (int)((source[t + 9] >> 2) | (source[t + 6] << 6));
            senselF = (int)(((source[t + 9] & 0x3) << 12) | (source[t + 8] << 4) | (source[t + 11] >> 4));
            senselG = (int)(((source[t + 11] & 0x0f) << 10) | (source[t + 10] << 2) | (source[t + 13] >> 6));
            senselH = (int)(((source[t + 13] & 0x3f) << 8) | (source[t + 12]));

            senselI = (int)((source[t + 14] >> 2) | (source[t + 15] << 6));
            senselJ = (int)(((source[t + 14] & 0x3) << 12) | (source[t + 17] << 4) | (source[t + 16] >> 4));
            senselK = (int)(((source[t + 16] & 0x0f) << 10) | (source[t + 19] << 2) | (source[t + 18] >> 6));
            senselL = (int)(((source[t + 18] & 0x3f) << 8) | (source[t + 21]));
            senselM = (int)((source[t + 23] >> 2) | (source[t + 20] << 6));
            senselN = (int)(((source[t + 23] & 0x3) << 12) | (source[t + 22] << 4) | (source[t + 25] >> 4));
            senselO = (int)(((source[t + 25] & 0x0f) << 10) | (source[t + 24] << 2) | (source[t + 27] >> 6));
            senselP = (int)(((source[t + 27] & 0x3f) << 8) | (source[t + 26]));

            senselQ = (int)((source[t + 28] >> 2) | (source[t + 29] << 6));
            senselR = (int)(((source[t + 28] & 0x3) << 12) | (source[t + 31] << 4) | (source[t + 30] >> 4));
            senselS = (int)(((source[t + 30] & 0x0f) << 10) | (source[t + 33] << 2) | (source[t + 32] >> 6));
            senselT = (int)(((source[t + 32] & 0x3f) << 8) | (source[t + 35]));
            senselU = (int)((source[t + 37] >> 2) | (source[t + 34] << 6));
            senselV = (int)(((source[t + 37] & 0x3) << 12) | (source[t + 36] << 4) | (source[t + 39] >> 4));
            senselW = (int)(((source[t + 39] & 0x0f) << 10) | (source[t + 38] << 2) | (source[t + 41] >> 6));
            senselX = (int)(((source[t + 41] & 0x3f) << 8) | (source[t + 40]));
            senselA = senselA >> 2;
            senselB = senselB >> 2;
            senselC = senselC >> 2;
            senselD = senselD >> 2;
            senselE = senselE >> 2;
            senselF = senselF >> 2;
            senselG = senselG >> 2;
            senselH = senselH >> 2;
            senselI = senselI >> 2;
            senselJ = senselJ >> 2;
            senselK = senselK >> 2;
            senselL = senselL >> 2;
            senselM = senselM >> 2;
            senselN = senselN >> 2;
            senselO = senselO >> 2;
            senselP = senselP >> 2;
            senselQ = senselQ >> 2;
            senselR = senselR >> 2;
            senselS = senselS >> 2;
            senselT = senselT >> 2;
            senselU = senselU >> 2;
            senselV = senselV >> 2;
            senselW = senselW >> 2;
            senselX = senselX >> 2;
        }

        Dest[tt++] = (unsigned char)((senselA >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselA & 0xF) << 4) | (senselB >> 8));
        Dest[tt++] = (unsigned char)(senselB & 0xff);

        Dest[tt++] = (unsigned char)((senselC >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselC & 0xF) << 4) | (senselD >> 8));
        Dest[tt++] = (unsigned char)(senselD & 0xff);

        Dest[tt++] = (unsigned char)((senselE >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselE & 0xF) << 4) | (senselF >> 8));
        Dest[tt++] = (unsigned char)(senselF & 0xff);

        Dest[tt++] = (unsigned char)((senselG >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselG & 0xF) << 4) | (senselH >> 8));
        Dest[tt++] = (unsigned char)(senselH & 0xff);

        Dest[tt++] = (unsigned char)((senselI >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselI & 0xF) << 4) | (senselJ >> 8));
        Dest[tt++] = (unsigned char)(senselJ & 0xff);

        Dest[tt++] = (unsigned char)((senselK >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselK & 0xF) << 4) | (senselL >> 8));
        Dest[tt++] = (unsigned char)(senselL & 0xff);

        Dest[tt++] = (unsigned char)((senselM >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselM & 0xF) << 4) | (senselN >> 8));
        Dest[tt++] = (unsigned char)(senselN & 0xff);

        Dest[tt++] = (unsigned char)((senselO >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselO & 0xF) << 4) | (senselP >> 8));
        Dest[tt++] = (unsigned char)(senselP & 0xff);

        Dest[tt++] = (unsigned char)((senselQ >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselQ & 0xF) << 4) | (senselR >> 8));
        Dest[tt++] = (unsigned char)(senselR & 0xff);

        Dest[tt++] = (unsigned char)((senselS >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselS & 0xF) << 4) | (senselT >> 8));
        Dest[tt++] = (unsigned char)(senselT & 0xff);

        Dest[tt++] = (unsigned char)((senselU >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselU & 0xF) << 4) | (senselV >> 8));
        Dest[tt++] = (unsigned char)(senselV & 0xff);

        Dest[tt++] = (unsigned char)((senselW >> 4) & 0xff);
        Dest[tt++] = (unsigned char)(((senselW & 0xF) << 4) | (senselX >> 8));
        Dest[tt++] = (unsigned char)(senselX & 0xff);

    }
    return destinationData[0];
}

