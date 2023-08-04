#ifndef GPU_NORM__
#define GPU_NORM__

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "timer.cuh"
#include "check.cuh"

template<typename T>
__global__ void init_one(T *init_mat, int size)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size)
        init_mat[idx] = 1;
}

template<typename T>
__inline__ __device__ void warpRecude(volatile T* s_y, int tid){
    s_y[tid] += s_y[tid + 32];
    s_y[tid] += s_y[tid + 16];
    s_y[tid] += s_y[tid + 8];
    s_y[tid] += s_y[tid + 4];
    s_y[tid] += s_y[tid + 2];
    s_y[tid] += s_y[tid + 1];
}


// one block for one sample
template<typename T>
__global__ void get_z_norm(T *Mat, const int MAT_ROWS, const int MAT_COLS, int ddof)
{
    const int N = 1024;
    __shared__ T sh_row[N];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if(tid<MAT_COLS)
        sh_row[tid] = Mat[bid*MAT_COLS+tid] * Mat[bid*MAT_COLS+tid];
    if(tid>MAT_COLS)
        sh_row[tid] = 0.0;
    __syncthreads();
    for (int i = tid+blockDim.x; i < MAT_COLS; i+=blockDim.x)
    {
        sh_row[tid] += Mat[bid*MAT_COLS+i] * Mat[bid*MAT_COLS+i];
    }
    __syncthreads();
    for (int i = blockDim.x>>1; i > 32; i >>= 1)
    {
        if(tid < i)
            sh_row[tid] += sh_row[tid+i];
        __syncthreads();
    }

     if(tid < 32){
        warpRecude(sh_row, tid);
    }
    
    if(tid == 0)
        sh_row[0] = sqrtf(sh_row[0]/(MAT_COLS - ddof));
    __syncthreads();
    for (int i = tid; i < MAT_COLS; i += blockDim.x)
    {
        Mat[bid*MAT_COLS+i] = Mat[bid*MAT_COLS+i]/sh_row[0];
    }
}


/*
//one thread for one sample
template<typename T>
__global__ void get_z_norm(T *Mat, const int MAT_ROWS,const int MAT_COLS, int ddof)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    
    if(idx < MAT_ROWS)
    {
        T temp_per_samp = 0;
        for (int i = 0; i < MAT_COLS; i++)
        {
            temp_per_samp += Mat[i + idx*MAT_COLS] * Mat[i + idx*MAT_COLS];
        }
        temp_per_samp = sqrtf(temp_per_samp/(MAT_COLS - ddof));
        for (int i = 0; i < MAT_COLS; i++)
        {
            Mat[i + idx*MAT_COLS] = Mat[i + idx*MAT_COLS]/temp_per_samp;
        }
    }
    
}
*/



template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type cub_excu_x(float *d_Mat, float *d_right_1, 
                                                      float *d_mean_result,
                                                      const int MAT_ROWS, const int MAT_COLS)
{
    cublasStatus_t stat;
    float alpha = 1.f/MAT_COLS, beta = 0;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {printf ("normalization CUBLAS initialization failed%d\n",stat);exit( -1 );}
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                       1, MAT_ROWS, MAT_COLS,          
                       &alpha, d_right_1, 1,          
                       d_Mat, MAT_COLS, &beta,             
                       d_mean_result, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {printf ("normalization CUBLAS cublasSgemm failed%d\n",stat);exit( -1 );}    
    alpha = -1;
    beta = 1;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                       MAT_COLS, MAT_ROWS, 1,          
                       &alpha, d_right_1, MAT_COLS,          
                       d_mean_result, 1, &beta,             
                       d_Mat, MAT_COLS);
    if (stat != CUBLAS_STATUS_SUCCESS) {printf ("normalization CUBLAS cublasSgemm failed%d\n",stat);exit( -1 );}  
    cublasDestroy(handle);
}

template <typename T>
typename std::enable_if<(sizeof(T) == 8)>::type cub_excu_x(double *d_Mat, double *d_right_1, 
                                                           double *d_mean_result,
                                                           const int MAT_ROWS, const int MAT_COLS)
{
    cublasStatus_t stat;
    double alpha = 1.f/MAT_COLS, beta = 0;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {printf ("normalization CUBLAS initialization failed%d\n",stat);exit( -1 );}
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                       1, MAT_ROWS, MAT_COLS,          
                       &alpha, d_right_1, 1,          
                       d_Mat, MAT_COLS, &beta,             
                       d_mean_result, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {printf ("normalization CUBLAS cublasDgemm failed%d\n",stat);exit( -1 );}    
    alpha = -1;
    beta = 1;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                MAT_COLS, MAT_ROWS, 1,          
                &alpha, d_right_1, MAT_COLS,          
                d_mean_result, 1, &beta,             
                d_Mat, MAT_COLS);
    if (stat != CUBLAS_STATUS_SUCCESS) {printf ("normalization CUBLAS cublasDgemm failed%d\n",stat);exit( -1 );}
    cublasDestroy(handle);
}


//行为样本
template<typename T>
void z_norm_gpu_x(T *d_Mat, const int MAT_ROWS, const int MAT_COLS, int ddof)
{
    /*
    int blockSize = 256;
    int gridsize = (MAT_ROWS + blockSize - 1) / blockSize;
    */
    int blockSize = 1024;
    int gridsize = MAT_ROWS;
    int gsize_right = (MAT_COLS + blockSize - 1) / blockSize;

    T *d_right_1;
    T *d_mean_result;

    CHECK(cudaMalloc((void**)&d_mean_result,sizeof(T)*MAT_ROWS));
    CHECK(cudaMalloc((void**)&d_right_1,sizeof(T)*MAT_COLS));
    
    init_one<<<gsize_right, blockSize>>>(d_right_1, MAT_COLS);
    cudaDeviceSynchronize();

    cub_excu_x<T>(d_Mat, d_right_1, d_mean_result, MAT_ROWS, MAT_COLS);
    
    get_z_norm<<<gridsize, blockSize>>>(d_Mat, MAT_ROWS, MAT_COLS, ddof);
    cudaDeviceSynchronize();
    
    cudaFree(d_mean_result);
    cudaFree(d_right_1);
    
}

#endif