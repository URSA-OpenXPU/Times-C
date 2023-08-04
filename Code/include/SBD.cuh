#ifndef GPU_SBD__
#define GPU_SBD__

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "ncc.cuh"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

template<typename T>
__device__ void zeroShift(T *d_vec, T *d_out, const int shift, const int vec_size)
{
    int abs_shift = shift>0?shift:(shift*(-1));
    
    if(shift == 0)
        memcpy(d_out, d_vec, sizeof(T)*vec_size);
    
    if(shift>0 && abs_shift<vec_size)
    {
        memcpy(d_out+shift, d_vec, sizeof(T)*(vec_size-shift));
    }
    if(shift<0 && abs_shift<vec_size)
    {
        memcpy(d_out, d_vec+abs_shift, sizeof(T)*(vec_size-abs_shift));
    }
}


template<typename T>
__device__ void get_maxpos(T *arr, const int arr_len, int &maxpos)
{
    T val = -1e10;
    for (int i = 0; i < arr_len; i++)
    {
        if(val < arr[i])
        {
             maxpos = i;
             val = arr[i];
        }    
    }
    
}


template<typename T>
__global__ void maxAndShift(T *d_mat_ncc, T *d_mat, T *d_mat_shift, 
                            const int ncc_len, const int mat_row, const int mat_col)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < mat_row)
    { 
        T *tmp_copy = (T*)malloc(sizeof(T)*mat_col);
        memset(tmp_copy, 0, sizeof(T)*mat_col);
        
        int pos = 0;
        get_maxpos(d_mat_ncc+ncc_len*idx, ncc_len, pos);
        
        zeroShift(d_mat+idx*mat_col, tmp_copy, (pos + 1)-mat_col, mat_col);
        memcpy(d_mat+idx*mat_col, tmp_copy, sizeof(T)*mat_col);
        free(tmp_copy);
    }
}


template<typename T>
void sbd3D(T *d_mat, T *d_center, T *d_mat_shift, 
          const int mat_row, const int mat_col)
{
    const int blockSize = 256;
    const int gridSize = (mat_row + blockSize - 1)/blockSize;
    T *d_mat_ncc;
    const int ncc_len = 2*mat_col-1;
    CHECK(cudaMalloc((void**)&d_mat_ncc, sizeof(T)*mat_row*ncc_len));

    NCC_3D(d_center, d_mat, d_mat_ncc, 1, mat_row, mat_col);
    
    maxAndShift<<<gridSize, blockSize>>>(d_mat_ncc, d_mat, d_mat_shift,
                                         ncc_len, mat_row, mat_col);
    cudaDeviceSynchronize();
    cudaFree(d_mat_ncc);
    
}

#endif