#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <vector>

#include "CycleTimer.h"
#include "crun.h"
#include "rutil.h"


#define BLOCK_SIZE 10




float toBW(int bytes, float sec) {
  return (float)(bytes) / (1024. * 1024. * 1024.) / sec;
}


__device__ __inline__ float mweight_kernel(float val, float optval) {
    float arg = 1.0 + COEFF * (val - optval);
    float lg = log(arg) * M_LOG2E;
    float denom = 1.0 + lg * lg;
    return 1.0/denom;
}

/* Compute imbalance between local and remote values */
/* Result < 0 when lcount > rcount and > 0 when lcount < rcount */
__device__ __inline__ float imbalance_kernel(int lcount, int rcount) {
    if (lcount == 0 && rcount == 0)
    return 0.0;
    float sl = sqrt((float) lcount);
    float sr = sqrt((float) rcount);
    return (sr-sl)/(sr+sl);
}
__device__ __inline__ float neighbor_ilf_kernel(int *rat_count, int *neighbor, int *neighbor_start, int nid) {
    int outdegree = neighbor_start[nid+1] - neighbor_start[nid] - 1;
    int *start = &neighbor[neighbor_start[nid]+1];
    int i;
    float sum = 0.0;
    for (i = 0; i < outdegree; i++) {
        int lcount = rat_count[nid];
        int rcount = rat_count[start[i]];
        float r = imbalance_kernel(lcount, rcount);
        sum += r;
    }
    float ilf = BASE_ILF + ILF_VARIABILITY * (sum/outdegree);
    return ilf;
}

__global__ void
compute_weight_kernel(float load_factor, int *rat_count, int *neighbor, int *neighbor_start, float* result, int width, int height) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_index = y * width + x; // thread_index is node id
    if (x < width && y < height){
        float ilf = neighbor_ilf_kernel(rat_count, neighbor, neighbor_start, thread_index);
        int count = rat_count[thread_index];
        result[thread_index] = mweight_kernel((float) count/load_factor, ilf);
    }
}


extern "C" void compute_all_weights_cuda(state_t *s){

    graph_t *g = s->g;
    int nnode = g->nnode;
    int nedge = g->nedge;
    int width = g->width;
    int height = g->height;
    int *rat_count_device;
    int *neighbor_device;
    int *neighbor_start_device; 
    float* result_device;
    printf("nnode %d", nnode);


    int totalBytes = sizeof(float) * nnode;

    
    double startTime = CycleTimer::currentSeconds();

    cudaMalloc(&rat_count_device, sizeof(int) * nnode);
    cudaMalloc(&neighbor_device, sizeof(int) * (nnode+nedge));
    cudaMalloc(&neighbor_start_device, sizeof(int) * (nnode+1));
    cudaMalloc(&result_device, sizeof(float) * nnode);

    cudaMemcpy(rat_count_device, s->rat_count, sizeof(int) * nnode, cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_device, g->neighbor, sizeof(int) * (nnode+nedge), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_start_device, g->neighbor_start,  sizeof(int) * (nnode+1), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    int num_block_x = (width+BLOCK_SIZE-1) / BLOCK_SIZE;
    int num_block_y = (height+BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(num_block_x, num_block_y, 1);

    double myTime = CycleTimer::currentSeconds();

    compute_weight_kernel<<<gridDim, blockDim>>>(s->load_factor, rat_count_device, neighbor_device, neighbor_start_device, result_device, width, height);
    cudaThreadSynchronize();
    double myTimeEnd = CycleTimer::currentSeconds();
    double time_without_mem = myTimeEnd - myTime;
    printf("Overall without memcpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * time_without_mem, toBW(totalBytes, time_without_mem));

    float *temp = (float *)malloc(sizeof(float) * nnode);

    cudaMemcpy(temp, result_device, sizeof(float) * nnode, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nnode; i++) {
        s->node_weight[i] = static_cast<double>(temp[i]);
    }
    cudaFree(rat_count_device);
    cudaFree(neighbor_device);
    cudaFree(neighbor_start_device);
    cudaFree(result_device);


    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    

}


// void
// saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

//     int totalBytes = sizeof(float) * 3 * N;

//     // compute number of blocks and threads per block
//     const int threadsPerBlock = 512;
//     const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

//     float* device_x;
//     float* device_y;
//     float* device_result;

//     //
//     // TODO allocate device memory buffers on the GPU using cudaMalloc
//     //

//     cudaMalloc(&device_x, sizeof(float) * N);
//     cudaMalloc(&device_y, sizeof(float) * N);
//     cudaMalloc(&device_result, sizeof(float) * N);


//     // start timing after allocation of device memory
//     double startTime = CycleTimer::currentSeconds();


//     //
//     // TODO copy input arrays to the GPU using cudaMemcpy
//     //
//     cudaMemcpy(device_x, xarray, sizeof(float) * N, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_y, yarray, sizeof(float) * N, cudaMemcpyHostToDevice);


//     // run kernel
//     double myTime = CycleTimer::currentSeconds();
//     saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
//     cudaThreadSynchronize();
//     double myTimeEnd = CycleTimer::currentSeconds();
//     double time_without_mem = myTimeEnd - myTime;
//     printf("Overall without memcpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * time_without_mem, toBW(totalBytes, time_without_mem));

//     //
//     // TODO copy result from GPU using cudaMemcpy
//     //
//     cudaMemcpy(resultarray, device_result, sizeof(float) * N, cudaMemcpyDeviceToHost);
//     printf("ooooo %f %f %f %f", alpha, xarray[1000], yarray[1000],resultarray[1000]);

//     // end timing after result has been copied back into host memory
//     double endTime = CycleTimer::currentSeconds();

//     cudaError_t errCode = cudaPeekAtLastError();
//     if (errCode != cudaSuccess) {
//         fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
//     }

//     double overallDuration = endTime - startTime;
//     printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

//     // TODO free memory buffers on the GPU
//     cudaFree(device_x);
//     cudaFree(device_y);
//     cudaFree(device_result);

// }

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
