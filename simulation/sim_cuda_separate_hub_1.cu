#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <vector>

#include "CycleTimer.h"
#include "crun.h"
#include "rutil.h"


#define BLOCK_SIZE 20
#define HUB_BLOCK_SIZE 128




float toBW(int bytes, float sec) {
  return (float)(bytes) / (1024. * 1024. * 1024.) / sec;
}


__device__ __inline__ double mweight_kernel(double val, double optval) {
    double arg = 1.0 + COEFF * (val - optval);
    double lg = log(arg) * M_LOG2E;
    double denom = 1.0 + lg * lg;
    return 1.0/denom;
}

/* Compute imbalance between local and remote values */
__device__ __inline__ double imbalance_density_kernel(double ldensity, double rdensity) {
    return (rdensity - ldensity) / (rdensity + ldensity);
}


__device__ __inline__ double neighbor_ilf_kernel(double load_factor, double *initial_load_factor, int *rat_count, int *infectious_rat_count, int *neighbor, int *neighbor_start, int nid, int max_outdegree) {
    int outdegree = neighbor_start[nid+1] - neighbor_start[nid] - 1;
    outdegree = min(outdegree, max_outdegree);
    int *start = &neighbor[neighbor_start[nid]+1];
    int i;
    double sum = 0.0;
    for (i = 0; i < outdegree; i++) {
       
        double ldensity = (rat_count[nid] == 0) ? 0.0 : 1.0 * infectious_rat_count[nid] / rat_count[nid];
        double rdensity = (rat_count[start[i]] == 0) ? 0.0 : 1.0 * infectious_rat_count[start[i]] / rat_count[start[i]];
        double r = (ldensity == 0.0 && rdensity == 0.0) ? 0.0 : imbalance_density_kernel(ldensity, rdensity);
        sum += r;
    }
    // change to a new ilf, where each node has different initial base ilf
    double ilf = BASE_ILF * (initial_load_factor[nid] / load_factor) + ILF_VARIABILITY * (sum / outdegree);
    return ilf;
}


__global__ void
compute_weight_hub_kernel(double load_factor, double *initial_load_factor, int* hub, int nhub, int *rat_count, int *infectious_rat_count, int *neighbor, int *neighbor_start, double* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < nhub) {
        int nodeid = hub[x];
        double ilf = neighbor_ilf_kernel(load_factor, initial_load_factor, rat_count, infectious_rat_count, neighbor, neighbor_start, nodeid, INT_MAX); // INT_MAX means compute for all possible neighbors
        int count = rat_count[nodeid];
        result[nodeid] = mweight_kernel((double) count/load_factor, ilf);
    }
}



__global__ void
compute_weight_kernel(bool *mask, double load_factor, double *initial_load_factor, int *rat_count, int *infectious_rat_count, int *neighbor, int *neighbor_start, double* result, int width, int height) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_index = y * width + x; // thread_index is node id

    if (x < width && y < height && mask[thread_index]){
        double ilf = neighbor_ilf_kernel(load_factor, initial_load_factor, rat_count, infectious_rat_count, neighbor, neighbor_start, thread_index, HUB_THREASHOLD); // 10 means only compute 4 neighbors
        int count = rat_count[thread_index];
        result[thread_index] = mweight_kernel((double) count/load_factor, ilf);
    }
}


extern "C" void init_cuda(state_t *s){
    return;
}
extern "C" void clean_cuda(){
    return;
}



extern "C" void compute_all_weights_cuda(state_t *s){

    graph_t *g = s->g;
    int nnode = g->nnode;
    int nedge = g->nedge;
    int width = g->width;
    int height = g->height;
    int nhub = g->nhub;

    int *hub_device;
    bool *mask_device;

    double *initial_load_factor_device;

    int *rat_count_device;
    int *infectious_rat_count_device;
    int *neighbor_device;
    int *neighbor_start_device; 
    double* weight_result_device;
    // printf("nhub, %d", nhub);

    int totalBytes = sizeof(double) * nnode;
    
    double startTime = CycleTimer::currentSeconds();

    cudaMalloc(&hub_device, sizeof(int) * nhub);
    cudaMalloc(&mask_device, sizeof(bool) * nnode);

    cudaMalloc(&rat_count_device, sizeof(int) * nnode);
    cudaMalloc(&infectious_rat_count_device, sizeof(int) * nnode);
    cudaMalloc(&neighbor_device, sizeof(int) * (nnode+nedge));
    cudaMalloc(&neighbor_start_device, sizeof(int) * (nnode+1));
    cudaMalloc(&weight_result_device, sizeof(double) * nnode);


    cudaMalloc(&initial_load_factor_device, sizeof(double) * nnode);

    cudaMemcpy(initial_load_factor_device, s->initial_load_factor, sizeof(double) * nnode, cudaMemcpyHostToDevice); 

    cudaMemcpy(hub_device, g->hub, sizeof(int) * nhub, cudaMemcpyHostToDevice);
    cudaMemcpy(mask_device, g->mask, sizeof(bool) * nnode, cudaMemcpyHostToDevice);
    cudaMemcpy(rat_count_device, s->rat_count, sizeof(int) * nnode, cudaMemcpyHostToDevice);
    cudaMemcpy(infectious_rat_count_device, s->infectious_rat_count, sizeof(int) * nnode, cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_device, g->neighbor, sizeof(int) * (nnode+nedge), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_start_device, g->neighbor_start,  sizeof(int) * (nnode+1), cudaMemcpyHostToDevice);



    double myTime = CycleTimer::currentSeconds();

    dim3 hubBlockDim(HUB_BLOCK_SIZE);
    int hub_num_block_x = (nhub + HUB_BLOCK_SIZE - 1) / HUB_BLOCK_SIZE;
    dim3 hubGridDim(hub_num_block_x);
    compute_weight_hub_kernel<<<hubGridDim, hubBlockDim>>>(s->load_factor, initial_load_factor_device, hub_device, nhub, rat_count_device,infectious_rat_count_device, neighbor_device, neighbor_start_device, weight_result_device);
    // cudaThreadSynchronize();
    // printf("Overall hub: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * (CycleTimer::currentSeconds()-myTime), toBW(totalBytes, (CycleTimer::currentSeconds()-myTime)));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    int num_block_x = (width+BLOCK_SIZE-1) / BLOCK_SIZE;
    int num_block_y = (height+BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(num_block_x, num_block_y, 1);
    compute_weight_kernel<<<gridDim, blockDim>>>(mask_device, s->load_factor, initial_load_factor_device, rat_count_device, infectious_rat_count_device, neighbor_device, neighbor_start_device, weight_result_device, width, height);

    cudaThreadSynchronize();
    double myTimeEnd = CycleTimer::currentSeconds();
    double time_without_mem = myTimeEnd - myTime;
    // printf("Overall without memcpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * time_without_mem, toBW(totalBytes, time_without_mem));



    cudaMemcpy(s->node_weight, weight_result_device, sizeof(double) * nnode, cudaMemcpyDeviceToHost);

    // double *hub_result_host = (double *) malloc(sizeof(double) * nhub);
    // cudaMemcpy(hub_result_host, hub_result_device, sizeof(double) * nhub, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < nhub; i++) {
    //     s->node_weight[g->hub[i]] = hub_result_host[i];
    // }

    // free(hub_result_host);


    cudaFree(hub_device);
    cudaFree(mask_device);

    cudaFree(rat_count_device);
    cudaFree(infectious_rat_count_device);
    cudaFree(neighbor_device);
    cudaFree(neighbor_start_device);
    cudaFree(weight_result_device);

    cudaFree(initial_load_factor_device);


    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    // printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    

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
