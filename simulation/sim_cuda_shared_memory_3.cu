#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <vector>

#include "CycleTimer.h"
#include "crun.h"
#include "rutil.h"


#define BLOCK_SIZE 16
#define HUB_BLOCK_SIZE 32

float toBW(int bytes, float sec) {
  return (float)(bytes) / (1024. * 1024. * 1024.) / sec;
}


int *hub_device;
bool *mask_device;
int *neighbor_device;
int *neighbor_start_device; 

double *initial_load_factor_device;
int *rat_count_device;
int *infectious_rat_count_device;
double* weight_result_device;

double *neighbor_accum_weight_result_device;
double *sum_weight_result_device;

extern "C" void init_cuda(state_t *s) {
    graph_t *g = s->g;
    int nnode = g->nnode;
    int nedge = g->nedge;
    int nhub = g->nhub;


    cudaMalloc(&hub_device, sizeof(int) * nhub);
    cudaMalloc(&mask_device, sizeof(bool) * nnode);
    cudaMalloc(&neighbor_device, sizeof(int) * (nnode+nedge));
    cudaMalloc(&neighbor_start_device, sizeof(int) * (nnode+1));

    cudaMalloc(&initial_load_factor_device, sizeof(double) * nnode);
    cudaMalloc(&rat_count_device, sizeof(int) * nnode);
    cudaMalloc(&infectious_rat_count_device, sizeof(int) * nnode);
    cudaMalloc(&weight_result_device, sizeof(double) * nnode);

    cudaMalloc(&neighbor_accum_weight_result_device, sizeof(double) * (nnode+nedge));
    cudaMalloc(&sum_weight_result_device, sizeof(double) * nnode);


    cudaMemcpy(initial_load_factor_device, s->initial_load_factor, sizeof(double) * nnode, cudaMemcpyHostToDevice); 
    cudaMemcpy(hub_device, g->hub, sizeof(int) * nhub, cudaMemcpyHostToDevice); 
    cudaMemcpy(mask_device, g->mask, sizeof(bool) * nnode, cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_device, g->neighbor, sizeof(int) * (nnode+nedge), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_start_device, g->neighbor_start,  sizeof(int) * (nnode+1), cudaMemcpyHostToDevice); 

}

extern "C" void clean_cuda() {
    cudaFree(hub_device);
    cudaFree(mask_device);
    cudaFree(neighbor_device);
    cudaFree(neighbor_start_device);

    cudaFree(initial_load_factor_device);
    cudaFree(rat_count_device);
    cudaFree(infectious_rat_count_device);
    cudaFree(weight_result_device);

    cudaFree(neighbor_accum_weight_result_device);
    cudaFree(sum_weight_result_device);
}





__device__ __inline__ double mweight_kernel(double val, double optval) {
    double arg = 1.0 + COEFF * (val - optval);
    double lg = log(arg) * M_LOG2E;
    double denom = 1.0 + lg * lg;
    return 1.0/denom;
}

/* Compute imbalance between local and remote values */
/* Result < 0 when lcount > rcount and > 0 when lcount < rcount */
__device__ __inline__ double imbalance_kernel(int lcount, int rcount) {
    if (lcount == 0 && rcount == 0)
    return 0.0;
    double sl = sqrt((double) lcount);
    double sr = sqrt((double) rcount);
    return (sr-sl)/(sr+sl);
}

__device__ __inline__ double imbalance_density_kernel(double ldensity, double rdensity) {
    return (rdensity - ldensity) / (rdensity + ldensity);
}


__device__ __inline__ double neighbor_ilf_fast_kernel(double load_factor, double *initial_load_factor, int *rat_count, int *infectious_rat_count, int* share_rat_count, int* share_infectious_rat_count, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nid = y * width + x; // thread_index is node id
    int in_block_x = threadIdx.x;
    int in_block_y = threadIdx.y;
    int in_block_id = in_block_x + in_block_y * BLOCK_SIZE;
    double sum = 0.0;
    double ldensity, rdensity;
    int remote_x, remote_y, remote_in_block_x, remote_in_block_y, remote_nid, remote_in_block_id;

    int outdegree = 4;
    if (x == 0) {
        outdegree--;
    }
    if (y == 0) {
        outdegree--;
    }
    if (x == width -1) {
        outdegree--;
    }
    if (y == height-1) {
        outdegree--;
    }

    ldensity = (share_rat_count[in_block_id] == 0) ? 0.0 : 1.0 * share_infectious_rat_count[in_block_id] / share_rat_count[in_block_id];
    //up
    remote_x = x;
    remote_y = y+1;
    remote_nid = remote_x + remote_y * width;
    remote_in_block_x = in_block_x;
    remote_in_block_y = in_block_y+1;
    remote_in_block_id = remote_in_block_x + remote_in_block_y * BLOCK_SIZE;

    if (remote_y < height) {
        if (remote_in_block_y >= BLOCK_SIZE) {
            rdensity = (rat_count[remote_nid] == 0) ? 0.0 : 1.0 * infectious_rat_count[remote_nid] / rat_count[remote_nid];
        } else {
            rdensity = (share_rat_count[remote_in_block_id] == 0) ? 0.0 : 1.0 * share_infectious_rat_count[remote_in_block_id] / share_rat_count[remote_in_block_id];
        }
        double r = (ldensity == 0.0 && rdensity == 0.0) ? 0.0 :imbalance_density_kernel(ldensity, rdensity);
        sum += r;
    }


   
    //down
    remote_x = x;
    remote_y = y-1;
    remote_nid = remote_x + remote_y * width;
    remote_in_block_x = in_block_x;
    remote_in_block_y = in_block_y-1;
    remote_in_block_id = remote_in_block_x + remote_in_block_y * BLOCK_SIZE;

    if (remote_y >= 0) {
        if (remote_in_block_y < 0) {
            rdensity = (rat_count[remote_nid] == 0) ? 0.0 : 1.0 * infectious_rat_count[remote_nid] / rat_count[remote_nid];
        } else {
            rdensity = (share_rat_count[remote_in_block_id] == 0) ? 0.0 : 1.0 * share_infectious_rat_count[remote_in_block_id] / share_rat_count[remote_in_block_id];
        }
        double r = (ldensity == 0.0 && rdensity == 0.0) ? 0.0 :imbalance_density_kernel(ldensity, rdensity);
        sum += r;
    }

    //left
    remote_x = x-1;
    remote_y = y;
    remote_nid = remote_x + remote_y * width;
    remote_in_block_x = in_block_x-1;
    remote_in_block_y = in_block_y;
    remote_in_block_id = remote_in_block_x + remote_in_block_y * BLOCK_SIZE;

    if (remote_x >= 0) {
        if (remote_in_block_x < 0) {
            rdensity = (rat_count[remote_nid] == 0) ? 0.0 : 1.0 * infectious_rat_count[remote_nid] / rat_count[remote_nid];
        } else {
            rdensity = (share_rat_count[remote_in_block_id] == 0) ? 0.0 : 1.0 * share_infectious_rat_count[remote_in_block_id] / share_rat_count[remote_in_block_id];
        }
        double r = (ldensity == 0.0 && rdensity == 0.0) ? 0.0 :imbalance_density_kernel(ldensity, rdensity);
        sum += r;
    }

    //right
    remote_x = x+1;
    remote_y = y;
    remote_nid = remote_x + remote_y * width;
    remote_in_block_x = in_block_x+1;
    remote_in_block_y = in_block_y;
    remote_in_block_id = remote_in_block_x + remote_in_block_y * BLOCK_SIZE;

    if (remote_x <width) {
        if (remote_in_block_x >= BLOCK_SIZE) {
            rdensity = (rat_count[remote_nid] == 0) ? 0.0 : 1.0 * infectious_rat_count[remote_nid] / rat_count[remote_nid];
        } else {
            rdensity = (share_rat_count[remote_in_block_id] == 0) ? 0.0 : 1.0 * share_infectious_rat_count[remote_in_block_id] / share_rat_count[remote_in_block_id];
        }
        double r = (ldensity == 0.0 && rdensity == 0.0) ? 0.0 :imbalance_density_kernel(ldensity, rdensity);
        sum += r;
    }

    // change to a new ilf, where each node has different initial base ilf
    double ilf = BASE_ILF * (initial_load_factor[nid] / load_factor) + ILF_VARIABILITY * (sum / outdegree);
    return ilf;
}


__device__ __inline__ double neighbor_ilf_hub_kernel(double load_factor, double *initial_load_factor, int *rat_count, int *infectious_rat_count, int *neighbor, int *neighbor_start, int nid, int max_outdegree) {
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
        double ilf = neighbor_ilf_hub_kernel(load_factor, initial_load_factor, rat_count, infectious_rat_count, neighbor, neighbor_start, nodeid, INT_MAX); // INT_MAX means compute for all possible neighbors
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

    int in_block_x = threadIdx.x;
    int in_block_y = threadIdx.y;
    int in_block_id = in_block_x + in_block_y * BLOCK_SIZE;


    __shared__ int share_rat_count[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int share_infectious_rat_count[BLOCK_SIZE * BLOCK_SIZE];

    share_rat_count[in_block_id] = rat_count[thread_index];
    share_infectious_rat_count[in_block_id] = infectious_rat_count[thread_index];



    if (x < width && y < height && mask[thread_index]){
        // double ilf = neighbor_ilf_hub_kernel(load_factor, initial_load_factor, rat_count, infectious_rat_count, neighbor, neighbor_start, thread_index, HUB_THREASHOLD); // INT_MAX means compute for all possible neighbors
        double ilf = neighbor_ilf_fast_kernel(load_factor, initial_load_factor, rat_count, infectious_rat_count, share_rat_count, share_infectious_rat_count, width, height); 
        int count = share_rat_count[in_block_id];
        result[thread_index] = mweight_kernel((double) count/load_factor, ilf);
    }
}



__global__ void find_all_sums_hub_kernel(int* hub, int nhub, double *node_weight, int *neighbor, int *neighbor_start, double *neighbor_accum_weight_result, double *sum_weight_result){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < nhub) {
        int nid = hub[x];
        double sum = 0.0;
        for (int eid = neighbor_start[nid]; eid < neighbor_start[nid+1]; eid++) { // this eid is just index of the neighbor in the neighbor array
            sum += node_weight[neighbor[eid]];
            neighbor_accum_weight_result[eid] = sum;
        }
        sum_weight_result[nid] = sum;
    }
}  
__global__ void find_all_sums_kernel(bool *mask, double *node_weight, int *neighbor, int *neighbor_start, double *neighbor_accum_weight_result, double *sum_weight_result, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nid = y * width + x; // thread_index is node id
    if (x < width && y < height && mask[nid]){
        double sum = 0.0;
        int end = min(neighbor_start[nid+1], neighbor_start[nid]+HUB_THREASHOLD+1); //+1 because HUB_THREASHOLD is out degree
        for (int eid = neighbor_start[nid]; eid < end; eid++) { // this eid is just index of the neighbor in the neighbor array
            sum += node_weight[neighbor[eid]];
            neighbor_accum_weight_result[eid] = sum;
        }
        sum_weight_result[nid] = sum;
    }
}   




extern "C" void compute_all_weights_cuda(state_t *s){

    graph_t *g = s->g;
    int nnode = g->nnode;
    int width = g->width;
    int height = g->height;
    int nhub = g->nhub;


    int totalBytes = sizeof(double) * nnode;
    
    double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(rat_count_device, s->rat_count, sizeof(int) * nnode, cudaMemcpyHostToDevice);
    cudaMemcpy(infectious_rat_count_device, s->infectious_rat_count, sizeof(int) * nnode, cudaMemcpyHostToDevice);


    double myTime = CycleTimer::currentSeconds();

    dim3 hubBlockDim(HUB_BLOCK_SIZE);
    int hub_num_block_x = (nhub + HUB_BLOCK_SIZE - 1) / HUB_BLOCK_SIZE;
    dim3 hubGridDim(hub_num_block_x);
    compute_weight_hub_kernel<<<hubGridDim, hubBlockDim>>>(s->load_factor, initial_load_factor_device, hub_device, nhub, rat_count_device,infectious_rat_count_device, neighbor_device, neighbor_start_device, weight_result_device);
    // cudaThreadSynchronize();
    // printf("compute_weights: Overall hub: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * (CycleTimer::currentSeconds()-myTime), toBW(totalBytes, (CycleTimer::currentSeconds()-myTime)));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    int num_block_x = (width+BLOCK_SIZE-1) / BLOCK_SIZE;
    int num_block_y = (height+BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(num_block_x, num_block_y, 1);
    compute_weight_kernel<<<gridDim, blockDim>>>(mask_device, s->load_factor, initial_load_factor_device, rat_count_device, infectious_rat_count_device, neighbor_device, neighbor_start_device, weight_result_device, width, height);

    cudaThreadSynchronize();
    double myTimeEnd = CycleTimer::currentSeconds();
    double time_without_mem = myTimeEnd - myTime;
    // printf("compute_weights: Overall without memcpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * time_without_mem, toBW(totalBytes, time_without_mem));



    cudaMemcpy(s->node_weight, weight_result_device, sizeof(double) * nnode, cudaMemcpyDeviceToHost);


    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    // printf("compute_weights: Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
}

extern "C" void find_all_sums_cuda(state_t *s){
    graph_t *g = s->g;
    int nnode = g->nnode;
    int nedge = g->nedge;
    int width = g->width;
    int height = g->height;
    int nhub = g->nhub;
    double startTime = CycleTimer::currentSeconds();
    // printf("nhub ooooo, %d, outdegree %d", nhub, g->neighbor_start[g->hub[0]+1] - g->neighbor_start[g->hub[0]] - 1);

    int totalBytes = sizeof(double) * nnode;



    dim3 hubBlockDim(HUB_BLOCK_SIZE);
    int hub_num_block_x = (nhub + HUB_BLOCK_SIZE - 1) / HUB_BLOCK_SIZE;
    dim3 hubGridDim(hub_num_block_x);
    find_all_sums_hub_kernel<<<hubGridDim, hubBlockDim>>>(hub_device, nhub, weight_result_device, neighbor_device, neighbor_start_device, neighbor_accum_weight_result_device, sum_weight_result_device);

    // cudaThreadSynchronize();
    // printf("find_sums: Overall normal node: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * (CycleTimer::currentSeconds()-startTime), toBW(totalBytes, (CycleTimer::currentSeconds()-startTime)));
    


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    int num_block_x = (width+BLOCK_SIZE-1) / BLOCK_SIZE;
    int num_block_y = (height+BLOCK_SIZE-1) / BLOCK_SIZE;
    dim3 gridDim(num_block_x, num_block_y, 1);
    find_all_sums_kernel<<<gridDim, blockDim>>>(mask_device, weight_result_device, neighbor_device, neighbor_start_device, neighbor_accum_weight_result_device, sum_weight_result_device, width, height);


    cudaThreadSynchronize();
    double myTimeEnd = CycleTimer::currentSeconds();
    double time_without_mem = myTimeEnd - startTime;
    // printf("find_sums: Overall without memcpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * time_without_mem, toBW(totalBytes, time_without_mem));

    cudaMemcpy(s->neighbor_accum_weight, neighbor_accum_weight_result_device, sizeof(double) * (nnode+nedge), cudaMemcpyDeviceToHost);
    cudaMemcpy(s->sum_weight, sum_weight_result_device, sizeof(double) * nnode, cudaMemcpyDeviceToHost);

    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    // printf("find_sums: Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

}



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
