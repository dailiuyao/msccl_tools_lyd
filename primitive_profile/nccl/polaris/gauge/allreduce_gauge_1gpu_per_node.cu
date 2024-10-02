#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

struct LogMessage_lyd* d_messages;

#if PROFILE_LYD_SEND_RECV_CHUNK == 1
std::chrono::time_point<std::chrono::high_resolution_clock> netIsend_time_start;
std::chrono::time_point<std::chrono::high_resolution_clock> netIrecv_time_start;
std::chrono::time_point<std::chrono::high_resolution_clock> netIsend_time_end;
std::chrono::time_point<std::chrono::high_resolution_clock> netIrecv_time_end;  
#endif

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

uint64_t rdtsc() {
    uint32_t lo, hi;
    // Inline assembly to read the TSC
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
}

int main(int argc, char* argv[])
{

  const char* env_gauge_heo_var = getenv("GAUGE_HEO");

  const char* env_gauge_mode_var = getenv("GAUGE_MODE");

  const char* env_gauge_iteration_var = getenv("GAUGE_ITERATION");

  const char* env_gauge_nchannels_var = getenv("GAUGE_NCHANNELS");

  const char* env_gauge_chunk_size_var = getenv("GAUGE_CHUNK_SIZE");

  const char* env_gauge_output_dir_var = getenv("GAUGE_OUT_DIRE");

  const char* env_gauge_nthreads_var = getenv("NCCL_NTHREADS");

  const char* env_comm_gpu_id_var = getenv("COMM_GPU_ID");

  // Check if environment variables are set
  if (!env_gauge_heo_var) env_gauge_heo_var = "unknown_gauge_heo";
  if (!env_gauge_mode_var) env_gauge_mode_var = "unknown_gauge_mode";
  if (!env_gauge_iteration_var) env_gauge_iteration_var = "unknown_gauge_iteration";
  if (!env_gauge_nchannels_var) env_gauge_nchannels_var = "unknown_gauge_nchannels";
  if (!env_gauge_chunk_size_var) env_gauge_chunk_size_var = "unknown_gauge_chunk_size";
  if (!env_gauge_nthreads_var) env_gauge_nthreads_var = "unknown_gauge_nthreads";  
  if (!env_gauge_output_dir_var) {
    env_gauge_output_dir_var = "unknown_gauge_output_dir";
    printf("unknown gauge output dir\n");
  }

  long long size = 1;  // Default size
  const char* env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
  if (env_gauge_size_var != nullptr) {
      size = atoll(env_gauge_size_var) * 1024 / 4;  // Convert from kilobytes to number of floats, assuming the environment variable is in kilobytes
  }

  const char* env_gauge_step_var = getenv("GAUGE_STEP_SIZE");

  int gauge_step = atoi(env_gauge_step_var);

  int comm_gpu_id = atoi(env_comm_gpu_id_var);

  int N_CHUNKS;

  // if (gauge_step != 0) {
  //   if (gauge_step >= 16384) {
  //     N_CHUNKS = 128;
  //   } else {
  //     N_CHUNKS = atoi(env_gauge_size_var)/atoi(env_gauge_step_var); 
  //   }
  // } else {
  //   N_CHUNKS = 1;
  // }

  // if (N_CHUNKS == 0) N_CHUNKS = 1;

  N_CHUNKS = MAXLOGLYD;

  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  char filename[256];

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  // select gpu on each node
  if (comm_gpu_id == 0) {
    localRank = localRank + 1;
  } else if (comm_gpu_id == 1){
    if (myRank == 0) localRank = localRank + 1;
  } else if (comm_gpu_id == 2){
    if (myRank == 1) localRank = localRank + 1;
  } else if (comm_gpu_id == 3){
    localRank = localRank + 3;
  } 


  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, N_ITERS * size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, N_ITERS * size * sizeof(float))); 
  CUDACHECK(cudaStreamCreate(&s));
  

  //gauge test
  CUDACHECK(cudaMalloc(&d_messages, sizeof(LogMessage_lyd)));
  CUDACHECK(cudaMemset(d_messages, 0, sizeof(LogMessage_lyd)));

  ////////////////////////////// PROFILE_LYD_P2P_DEVICE_SYNC: START //////////////////////////////
  
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //communicating using NCCL

  cudaEvent_t start, stop;
  float elapsed_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  #if PROFILE_LYD_SEND_RECV_CHUNK == 1 
  netIsend_time_start = std::chrono::time_point<std::chrono::high_resolution_clock>();
  netIrecv_time_start = std::chrono::time_point<std::chrono::high_resolution_clock>();
  netIsend_time_end = std::chrono::time_point<std::chrono::high_resolution_clock>();
  netIrecv_time_end = std::chrono::time_point<std::chrono::high_resolution_clock>();
  #endif

  CUDACHECK(cudaStreamSynchronize(s));

  cudaEventRecord(start, s);

  std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_start_time = std::chrono::high_resolution_clock::now(); 

  CUDACHECK(cudaStreamSynchronize(s));

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));

  CUDACHECK(cudaStreamSynchronize(s));

  cudaEventRecord(stop, s);

  std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_end_time = std::chrono::high_resolution_clock::now();

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate elapsed time between events
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // Destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop); 

  #if PROFILE_LYD_SEND_RECV_CHUNK == 1 
  std::chrono::duration<float, std::milli> func_netIsend_time = netIsend_time_start - nccl_func_start_time; 

  std::chrono::duration<float, std::milli> netIsend_total_time = netIsend_time_end - netIsend_time_start;  

  std::chrono::duration<float, std::milli> netIrecv_total_time = netIrecv_time_end - netIrecv_time_start;  

  std::chrono::duration<float, std::milli> netIrecv_func_time = nccl_func_end_time - netIrecv_time_end; 
  #endif
  
  std::chrono::duration<float, std::milli> nccl_func_time = nccl_func_end_time - nccl_func_start_time; 

  ////////////////////////////// PROFILE_LYD_P2P_DEVICE_SYNC: END //////////////////////////////

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  if (myRank < 4) {
    sprintf(filename, "%s/nccl_allreduce_%s_r-%d.out", env_gauge_output_dir_var, env_gauge_heo_var, myRank);
    freopen(filename, "a", stdout);
  } else {
    freopen("/dev/null", "w", stdout);
  }

  // After the kernel execution, copy the messages back to the host
  LogMessage_lyd* h_messages = new LogMessage_lyd;
  cudaMemcpy(h_messages, d_messages, sizeof(LogMessage_lyd), cudaMemcpyDeviceToHost);

  #if PROFILE_LYD_REDUCE_BROADCAST == 1
  double gauge_time;
  
  if (myRank == 0) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    gauge_time = static_cast<double>(h_messages->timeValue[0][1] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("rrc_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
  } else if (myRank == 1) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    gauge_time = static_cast<double>(h_messages->timeValue[0][1] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("send_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
  } else if (myRank == 2) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    gauge_time = static_cast<double>(h_messages->timeValue[0][1] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("rrs_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
  } else if (myRank == 3) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    gauge_time = static_cast<double>(h_messages->timeValue[0][1] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("send_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, env_gauge_iteration_var, gauge_time);
  }
  #endif

  #if PROFILE_LYD_REDUCE_BROADCAST_CHUNK == 1
  double gauge_time;
  
  if (myRank == 0) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    for (size_t i = 0; i < N_CHUNKS; ++i) { 
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
      printf("rrc_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
      printf("send_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    }
  } else if (myRank == 1) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    for (size_t i = 0; i < N_CHUNKS; ++i) { 
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
      printf("send_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
      printf("recv_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    }
  } else if (myRank == 2) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    for (size_t i = 0; i < N_CHUNKS; ++i) { 
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
      printf("rrs_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
      printf("rcs_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    }
  } else if (myRank == 3) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
    for (size_t i = 0; i < N_CHUNKS; ++i) { 
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
      printf("send_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
      printf("recv_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    }
  }
  #endif

  #if PROFILE_LYD_RS_AG_CHUNK == 1
  double gauge_time;
  
  printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, elapsed_time);
  printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_iteration(%s)_nccl allreduce elapsed time by clock: %.3f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, env_gauge_iteration_var, nccl_func_time.count());
  for (size_t i = 0; i < N_CHUNKS; ++i) { 
    gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][i]) / GAUGE_GPU_FREQUENCY;
    printf("send_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[3][i] - h_messages->timeValue[2][i]) / GAUGE_GPU_FREQUENCY;
    printf("rrs_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[5][i] - h_messages->timeValue[4][i]) / GAUGE_GPU_FREQUENCY;
    printf("rrcs_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[7][i] - h_messages->timeValue[6][i]) / GAUGE_GPU_FREQUENCY;
    printf("rcs_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[9][i] - h_messages->timeValue[8][i]) / GAUGE_GPU_FREQUENCY;
    printf("recv_heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, env_gauge_iteration_var, gauge_time);
  }
  
  #endif

  // Free the device memory of the gauge test
  cudaFree(d_messages);
  delete[] h_messages;


  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}