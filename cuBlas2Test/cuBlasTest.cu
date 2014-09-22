#include <stdio.h>
#include <cublas_v2.h>
#define M 16
#define N 128
#define K 512


int main(){

  float *d_a, *d_b, *d_c;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaMalloc(&d_b, K*M*sizeof(float));
  cudaMalloc(&d_c, N*M*sizeof(float));
  cudaMalloc(&d_a, N*K*sizeof(float));

  cublasHandle_t my_handle;
  cublasStatus_t my_status = cublasCreate(&my_handle);
  if (my_status != CUBLAS_STATUS_SUCCESS) {printf("handle failure %d\n", (int)my_status); return 1;}
  cudaMemset(d_a, 0, N*K*sizeof(float));
  cudaMemset(d_b, 0, K*M*sizeof(float));

  my_status = cublasSgemm(my_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_b, M, d_a, K, &beta, d_c, M);
  if (my_status != CUBLAS_STATUS_SUCCESS) {printf("Sgemm failure %d\n", (int)my_status); return 1;}
  printf("Success\n");
  return 0;
}
