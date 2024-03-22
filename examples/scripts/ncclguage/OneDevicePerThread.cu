#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

struct LogMessage_lyd* d_messages;

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


int main(int argc, char* argv[])
{
  int size = 32*1024*1024;


  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


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


  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));
  

  //gauge test
  CUDACHECK(cudaMalloc(&d_messages, sizeof(LogMessage_lyd)));
  CUDACHECK(cudaMemset(d_messages, 0, sizeof(LogMessage_lyd)));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));


  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  // After the kernel execution, copy the messages back to the host
  LogMessage_lyd* h_messages = new LogMessage_lyd;
  cudaMemcpy(h_messages, d_messages, sizeof(LogMessage_lyd), cudaMemcpyDeviceToHost);

  // Process and print the messages on the host
  #if PROFILE_LYD_REDUCE_BROADCAST == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | allreduce.h | runTreeUpDown | recvReduceCopy | time: %f us\n", h_messages->timeValue[i][0]);
  }

  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | allreduce.h | runTreeUpDown | directSendFromOutput | time: %f us\n", h_messages->timeValue1[i][0]);
  }
  #endif

  #if PROFILE_LYD_REDUCE_BROADCAST_CHUNK == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | allreduce.h | runTreeUpDown | recvReduceCopy-chunk | iteration %d | time: %f us\n", j, h_messages->timeValue[i][j]);
    }
  }
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | allreduce.h | runTreeUpDown | directSendFromOutput-chunk | iteration %d | time: %f us\n", j, h_messages->timeValue1[i][j]);
    }
  }
  #endif

  #if PROFILE_LYD_REDUCE_LOADCONN_SETDATA == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | primitives | loadRecvConn | time: %f us\n", h_messages->timeValue[i][0]);
  }

  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | primitives | loadSendConn | time: %f us\n", h_messages->timeValue1[i][0]);
  }

  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | primitives | setDataPtrs | time: %f us\n", h_messages->timeValue2[i][0]);
  }
  #endif

  #if PROFILE_LYD_GENERIC == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | genericop | time: %f us\n", h_messages->timeValue[i][0]);
  }
  #endif

  #if PROFILE_LYD_WAIT_REDUCE_COPY_POST
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | prims_simple.h | genericOp | waitpeer | iteration %d | time: %f us\n", j, h_messages->timeValue[i][j]);
    }
  }
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | prims_simple.h | genericOp | ReduceOrCopyMulti | iteration %d | time: %f us\n", j, h_messages->timeValue1[i][j]);
    }
  }
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | prims_simple.h | genericOp | postPeer | iteration %d | time: %f us\n", j, h_messages->timeValue2[i][j]);
    }
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