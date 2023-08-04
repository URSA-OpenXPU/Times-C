#ifndef GPU_CENTER__
#define GPU_CENTER__

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "SBD.cuh"
#include "normalization.cuh"
#include "timer.cuh"
#include "check.cuh"
#include "print_test.cuh"
#include "readfile.cuh"

#define TILE_WIDTH 16


template<typename T>
__global__ void class_count(T *d_idx, const int idx_len, T *k_count)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < idx_len)
    {
        atomicAdd(&k_count[d_idx[idx] - 1], 1);
    }
}

template<typename T>
__global__ void histo_split(T *d_idx, T *count_tmp,
                           const int step_size, const int k, const int length)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    int step = step_size;
    if(idx*step_size < length && (idx+1)*step_size > length)
        step = length%step_size;
    if(idx*step_size < length)
    {
        T *cur_count = (T*)malloc(sizeof(T)*k);
        memset(cur_count, 0, sizeof(T)*k);
        for (int i = 0; i < step; i++)
        {
            cur_count[d_idx[idx*step_size+i] - 1]++;
        }
        for (int i = 0; i < k; i++)
        {
            count_tmp[stride*i + idx] = cur_count[i];
            
        }
        
        free(cur_count);
    }
}


template<typename T>
__device__ void sum_reduce(T *arr, const int arr_size, T &sum)
{
    for (int i = 0; i < arr_size; i++)
    {
        sum += arr[i];
    }
    
}

template<typename T>
__global__ void histo_merge(T *d_idx, T *count_tmp, T *d_kcount,
                           const int step_size, const int k, const int length)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    if(idx < k)
    {
        sum_reduce(count_tmp+idx*stride, stride, d_kcount[idx]);
    }
}

// two levels hash
template<typename T>
__global__ void get_loc2(T *d_idx, T *count_k_cur, T *loc2, const int idx_len)
{
    for (int i = 0; i < idx_len; i++)
    {
        loc2[i] = count_k_cur[d_idx[i] - 1];
        count_k_cur[d_idx[i] - 1]++;
    } 
}

template<typename T>
__global__ void loc2_split(T *d_idx, T *d_count_tmp, T *d_loc2,
                           const int step_size, const int k, const int length)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    int step = step_size;
    if(idx*step_size < length && (idx+1)*step_size > length)
        step = length%step_size;
    if(idx*step_size < length)
    {
        T *cur_count = (T*)malloc(sizeof(T)*k);
        memset(cur_count, 0, sizeof(T)*k);
        for (int i = 0; i < step; i++)
        {
            d_loc2[idx*step_size+i] = cur_count[d_idx[idx*step_size+i] - 1];
            cur_count[d_idx[idx*step_size+i] - 1]++;
        }

        //map cur_count to count_tmp
        for (int i = 0; i < k; i++)
        {
            d_count_tmp[i*stride+idx] = cur_count[i];
        }
        free(cur_count);
    }
    
}

template<typename T>
__device__ void exclusive_scan(T *arr, const int arr_size)
{
    T tmp_cur, tmp_next;
    tmp_cur = arr[0];
    arr[0] = 0;
    for (int i = 1; i < arr_size; i++)
    {
        tmp_next = arr[i];
        arr[i] = tmp_cur + arr[i - 1];
        tmp_cur = tmp_next;
    }
}

template<typename T>
__global__ void per_scan(T *d_count_tmp, const int step_size, const int k, const int length)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    if(idx < k)
    {
        //thrust::exclusive_scan(thrust::device, d_count_tmp+idx*stride, 
        //                       d_count_tmp+(idx+1)*stride, d_count_tmp+idx*stride);
        exclusive_scan(d_count_tmp+idx*stride, stride);
    }
}

template<typename T>
__global__ void loc2_merge(T *d_idx, T *d_count_tmp, T *d_loc2,
                           const int step_size, const int k, const int length)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    int step = step_size;
    if(idx*step_size < length && (idx+1)*step_size > length)
        step = length%step_size;
    if(idx*step_size < length)
    {
        T *cur_count = (T*)malloc(sizeof(T)*k);
        memset(cur_count, 0, sizeof(T)*k);
        //map count_tmp to cur_count
        for (int i = 0; i < k; i++)
        {
            cur_count[i] = d_count_tmp[i*stride+idx];
        }

        for (int i = 0; i < step; i++)
        {
            d_loc2[idx*step_size+i] += cur_count[d_idx[idx*step_size+i] - 1];
        }
        free(cur_count);
    }
}

// hash sort --- using 2D blocks and threads
template<typename T>
__global__ void block_alloc2D(T *d_mat_seg, const T *d_mat, int *d_idx, int *d_scan_kcount,
                              int *d_loc2, const int mat_row, const int mat_col)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int idy = threadIdx.y + blockDim.y*blockIdx.y;
    if (idx < mat_row && idy < mat_col)
    {
        const int pos0 = d_idx[idx] - 1;
        const int pos1 = d_scan_kcount[pos0]*mat_col + mat_col*d_loc2[idx];
        d_mat_seg[pos1 + idy] = d_mat[idx*mat_col+idy];
    }
    
}

//copy first sample to check eigenvector pos or neg
template<typename T>
__global__ void val_copy(T *d_mat_seg, T *d_mat_val, int *d_scan_kcount,
                         const int k, const int mat_col)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < k)
    {
        memcpy(d_mat_val+idx*mat_col, d_mat_seg+d_scan_kcount[idx]*mat_col, sizeof(T)*mat_col);
    }
}

//matrix Q 
template<typename T>
__global__ void get_matQ(T *d_matQ, const int mat_row, const int mat_col)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int idy = threadIdx.y + blockDim.y*blockIdx.y;
    if(idx < mat_row && idy < mat_col)
    {
        T a = (-1)*(1.0/mat_col);
        T b = 1 - 1.0/mat_col;
        d_matQ[idx + idy*mat_col] = idx == idy ? b:a;
    }
}

template<typename T>
__device__ void ops_center(T *d_centers, const int MAT_COLS)
{
    for (int i = 0; i < MAT_COLS; i++)
    {
         d_centers[i] = -1 * d_centers[i];
    }
}

//judge eigenvector pos or neg
template<typename T>
__device__ void judge_center(T *d_vals, T *d_centers, const int MAT_COLS, 
                             bool flag, T &finddistance)
{
    if(flag)
    {
        for (int idx = 0; idx < MAT_COLS; idx++)
        {
            finddistance += (d_vals[idx] - d_centers[idx])*(d_vals[idx] - d_centers[idx]); 
        }
        finddistance = sqrtf(finddistance);    
    }
    else
    {
        for (int idx = 0; idx < MAT_COLS; idx++)
        {
            finddistance += (d_vals[idx] + d_centers[idx])*(d_vals[idx] + d_centers[idx]);
        }
        finddistance = sqrtf(finddistance);     
    }
     
}

template<typename T>
__global__ void get_center(T *d_vals, T *d_EigVecs, T *d_centers, int *d_scan_kcount,
                          const int k, const int MAT_COLS, int *d_c)
{
    
    T finddistance1 = 0;
    T finddistance2 = 0;
    judge_center(d_vals+MAT_COLS*d_c[0], d_EigVecs, MAT_COLS, true, finddistance1);
    judge_center(d_vals+MAT_COLS*d_c[0], d_EigVecs, MAT_COLS, false, finddistance2);
    if(finddistance1 >= finddistance2)
    {
        ops_center(d_EigVecs, MAT_COLS);
    }
        
    memcpy(d_centers+d_c[0]*MAT_COLS, d_EigVecs, sizeof(T)*MAT_COLS);
}


//one block -- one vector
template<typename T>
__global__ void init_vec(T *d_vec, const int MAT_COLS)
{
    const int bsize = blockDim.x;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    for (int i = tid; i < MAT_COLS; i+=bsize)
    {
        
        if(i%2 == 0)
            d_vec[bid*MAT_COLS+i] = 1.00;
        if(i%2 == 1)
            d_vec[bid*MAT_COLS+i] = 0.50;
    }
    
}


template<typename T>
__global__ void norm_vec(T *d_vec, const int MAT_COLS)
{
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int bsize = blockDim.x;
    const int tid = threadIdx.x;
    //const int bid = blockIdx.x;
    __shared__ T vec_square[1024];

    if(tid < MAT_COLS)
        vec_square[tid] = d_vec[tid]*d_vec[tid];
        
    if(tid >= MAT_COLS)
        vec_square[tid] = 0.0;
    __syncthreads();
    for (int i = tid+bsize; i < MAT_COLS; i+=bsize)
    {
        vec_square[tid] += d_vec[i]*d_vec[i];
    }
    __syncthreads();
    for (int i = blockDim.x>>1; i > 32; i >>= 1)
    {
        if(tid < i)
            vec_square[tid] += vec_square[tid+i];
        __syncthreads();
    }

    if(tid < 32){
        warpRecude(vec_square, tid);
    }

    if(tid == 0)
        vec_square[0] = sqrtf(vec_square[0]);
    __syncthreads();

    if(idx < MAT_COLS)
        d_vec[idx] /= vec_square[0];
}


template<typename T>
__global__ void dot_reduce(T *d_vec, T *d_lamdavec, const int MAT_COLS, T *d_lamda)
{
    const int bsize = blockDim.x;
    const int tid = threadIdx.x;
    //const int bid = blockIdx.x;
    __shared__ T vec_dot[1024];
    for (int i = tid; i < 1024; i+=bsize)
    {
        vec_dot[i] = 0;
    }
    if(tid < MAT_COLS)
        vec_dot[tid] = d_vec[tid]*d_lamdavec[tid];
    __syncthreads();
    for (int i = tid+bsize; i < MAT_COLS; i+=bsize)
    {
        vec_dot[tid] += d_vec[i]*d_lamdavec[i];
    }
    __syncthreads();
    for (int i = blockDim.x>>1; i > 32; i >>= 1)
    {
        if(tid < i)
            vec_dot[tid] += vec_dot[tid+i];
        __syncthreads();
    }

    if (tid < 32)
    {
        warpRecude(vec_dot, tid);
    }
    
    if(tid == 0)
        d_lamda[0] = vec_dot[0];

}





template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type get_class_centers_a(float *d_mat_seg, float *d_vals, 
                                                                  float *d_centers,
                                                                  int *d_scan_kcount, int *scan_kcount, 
                                                                  int *k_count, const int k, 
                                                                  const int MAT_COLS)
{
    
    float alpha = 1.0, beta = 0.0;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((MAT_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (MAT_COLS+threadsPerBlock.y-1)/threadsPerBlock.y);   

    T *d_matQ;
    CHECK(cudaMalloc((void**)&d_matQ, sizeof(T)*MAT_COLS*MAT_COLS));
    get_matQ<<<numBlocks, threadsPerBlock>>>(d_matQ, MAT_COLS, MAT_COLS);
    cudaDeviceSynchronize();

    float *d_result;
    CHECK(cudaMalloc((void**)&d_result, sizeof(float)*MAT_COLS*MAT_COLS)); 
    cudaMemset(d_result, 0, sizeof(float)*MAT_COLS*MAT_COLS);


    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            cublasHandle_t handle;
            cusolverDnHandle_t cusolverH;
            cublasStatus_t blasStat = cublasCreate(&handle);
            cusolverStatus_t solverStat = cusolverDnCreate(&cusolverH);

            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("shapeExtract CUBLAS initialization failed%d\n",blasStat);exit( -1 );}
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("shapeExtract CUSOLVER initialization failed%d\n",solverStat);exit( -1 );}
    
            blasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,   
                                   MAT_COLS, MAT_COLS, k_count[i],          
                                   &alpha, d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS,          
                                   d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS, &beta,             
                                   d_result, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
            

            //Q*S
            blasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                                   MAT_COLS, MAT_COLS, MAT_COLS,          
                                   &alpha, d_result, MAT_COLS,          
                                   d_matQ, MAT_COLS, &beta,             
                                   d_result, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}

            //(Q*S)*Q
            blasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                        MAT_COLS, MAT_COLS, MAT_COLS,          
                        &alpha, d_matQ, MAT_COLS,          
                        d_result, MAT_COLS, &beta,             
                        d_result, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}

            float *d_EigVal; 
            float *d_work;
            int *devInfo; 
	        int lwork = 0;

            CHECK(cudaMalloc((void**)&d_EigVal, sizeof(float)*MAT_COLS));
            CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
            
	        solverStat = cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, MAT_COLS, 
                                                     d_result, MAT_COLS, 
                                                     d_EigVal, &lwork);

            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set buffer failed%d\n",solverStat);exit( -1 );}
            
	        CHECK(cudaMalloc((void**)&d_work, sizeof(float)*lwork));

            
            solverStat = cusolverDnSsyevd(cusolverH, jobz, uplo, MAT_COLS, 
                                          d_result, MAT_COLS, 
                                          d_EigVal, d_work, lwork, devInfo);
            
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER cusolverDnSsyevd failed%d\n",solverStat);exit( -1 );}
            
            cudaFree(d_EigVal);
            cudaFree(d_work);
            cudaFree(devInfo);
            cublasDestroy(handle);
            cusolverDnDestroy(cusolverH);

            int c[1];
            c[0] = i;
            int *d_c;
            CHECK(cudaMalloc((void**)&d_c, sizeof(int)));
            cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);
            get_center<<<1, 1>>>(d_vals, d_result+(MAT_COLS-1)*MAT_COLS, d_centers, d_scan_kcount, k, MAT_COLS, d_c);
            cudaDeviceSynchronize();
            
        }
        
    }   
    cudaFree(d_matQ);
    cudaFree(d_result);
}




//calcute eigenvector to get centrious
template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type get_class_centers(float *d_mat_seg, float *d_vals, 
                                                                  float *d_centers,
                                                                  int *d_scan_kcount, int *scan_kcount, 
                                                                  int *k_count, const int k, 
                                                                  const int MAT_COLS,
                                                                  int &sum_classes)
{
    
    cudaStream_t *streams = (cudaStream_t*)malloc(k*sizeof(cudaStream_t));
    cublasHandle_t *handle = (cublasHandle_t*)malloc(k*sizeof(cublasHandle_t));
    
    cublasStatus_t blasStat;
    cudaError_t streamStat;
    
    sum_classes = 0;
    int *classes_scan = (int*)malloc(k*sizeof(int));
    classes_scan[0] = 0;
    for (int i = 0; i < k-1; i++)
    {
        if(k_count[i] != 0)
            classes_scan[i+1] = classes_scan[i]+1;
        else
            classes_scan[i+1] = classes_scan[i];
    }
    
    for(int i=0; i<k; i++)
    {
         blasStat = cublasCreate(&handle[i]);
         if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("shapeExtract CUBLAS initialization failed%d\n",blasStat);exit( -1 );}
         streamStat = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
         if (streamStat != cudaSuccess) {printf ("shapeExtract cudaStream initialization failed%d\n",streamStat);exit( -1 );}
         if(k_count[i] != 0)
            sum_classes += 1;
    }

    printf("sum_class: %d\n", sum_classes);

    float *d_result;
    CHECK(cudaMalloc((void**)&d_result, sizeof(float)*sum_classes*MAT_COLS*MAT_COLS)); 
    cudaMemset(d_result, 0, sizeof(float)*sum_classes*MAT_COLS*MAT_COLS);

        
    float alpha = 1.0, beta = 0.0;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((MAT_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (MAT_COLS+threadsPerBlock.y-1)/threadsPerBlock.y);   

    for(int i=0; i<k; i++)
    {
        blasStat = cublasSetStream(handle[i], streams[i]);
        if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS stream set failed%d\n",blasStat);exit( -1 );}

    }

    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            blasStat = cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_T,   
                                   MAT_COLS, MAT_COLS, k_count[i],          
                                   &alpha, d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS,          
                                   d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS, &beta,             
                                   d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
        }
    }   
            
    T *d_matQ;
    CHECK(cudaMalloc((void**)&d_matQ, sizeof(T)*MAT_COLS*MAT_COLS));
    get_matQ<<<numBlocks, threadsPerBlock>>>(d_matQ, MAT_COLS, MAT_COLS);
    cudaDeviceSynchronize();
    
    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            //Q*S
            blasStat = cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                                   MAT_COLS, MAT_COLS, MAT_COLS,          
                                   &alpha, d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS,          
                                   d_matQ, MAT_COLS, &beta,             
                                   d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
        }
    }

    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            //(Q*S)*Q
            blasStat = cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                        MAT_COLS, MAT_COLS, MAT_COLS,          
                        &alpha, d_matQ, MAT_COLS,          
                        d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS, &beta,             
                        d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
        }
    }
    
    cudaFree(d_matQ);


    float *d_iterv;
    CHECK(cudaMalloc((void**)&d_iterv, sizeof(float)*sum_classes*MAT_COLS));  
    int initvecThreads = 512;
    int initvecBlocks = sum_classes;
    //one block -- one vector   
    init_vec<float><<<initvecBlocks, initvecThreads>>>(d_iterv, MAT_COLS);
    cudaDeviceSynchronize();


    const int normThreads = 1024;
    const int normBlocks = (MAT_COLS+normThreads-1)/normThreads;
    //const int uninormBlocks = sum_classes;
    
    for(int j = 0; j < MAT_COLS; j++)
    {
        for (int i=0; i<k; i++)
        {
            if(k_count[i] != 0)
            {
                cublasSgemv(handle[i], CUBLAS_OP_T, MAT_COLS, MAT_COLS,
                        &alpha, d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS,
                        d_iterv+classes_scan[i]*MAT_COLS, 1, &beta, 
                        d_iterv+classes_scan[i]*MAT_COLS, 1);
                
                norm_vec<float><<<normBlocks, normThreads, 0, streams[i]>>>
                                    (d_iterv+classes_scan[i]*MAT_COLS, MAT_COLS);
                cudaDeviceSynchronize();            
            }
        }
    }


    float *d_lamdavec;
    float *d_lamda;
    float *h_lamda = (float *) malloc(sizeof(float)*k);
    CHECK(cudaMalloc((void**)&d_lamdavec, sizeof(float)*sum_classes*MAT_COLS));
    CHECK(cudaMalloc((void**)&d_lamda, sizeof(float)*k));
    CHECK(cudaMemset(d_lamda, 0, sizeof(float)*k));
    CHECK(cudaMemcpy(d_lamdavec, d_iterv, sizeof(float)*sum_classes*MAT_COLS, cudaMemcpyDeviceToDevice));
    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            cublasSgemv(handle[i], CUBLAS_OP_T, MAT_COLS, MAT_COLS,
                        &alpha, d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS,
                        d_lamdavec+classes_scan[i]*MAT_COLS, 1, 
                        &beta, d_lamdavec+classes_scan[i]*MAT_COLS, 1);
            dot_reduce<float><<<1, normThreads, 0, streams[i]>>>
                    (d_iterv+classes_scan[i]*MAT_COLS, d_lamdavec+classes_scan[i]*MAT_COLS,
                     MAT_COLS, d_lamda+i);
            cudaDeviceSynchronize();
        }
    }
    CHECK(cudaMemcpy(h_lamda, d_lamda, sizeof(float)*sum_classes, cudaMemcpyDeviceToHost));
    cudaFree(d_lamdavec);
    cudaFree(d_lamda);
    

    for(int i=0; i<k; i++)
    {
        
        if(k_count[i] != 0 && h_lamda[i] > 0)
        {
            int c[1];
            c[0] = i;
            int *d_c;
            CHECK(cudaMalloc((void**)&d_c, sizeof(int)));
            cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);
            get_center<<<1, 1>>>(d_vals, d_iterv+classes_scan[i]*MAT_COLS, d_centers, 
                                 d_scan_kcount, k, MAT_COLS, d_c);
            cudaDeviceSynchronize();
        }
        if(k_count[i] != 0 && h_lamda[i] <= 0)
        {
            cusolverDnHandle_t cusolverH;
            cusolverStatus_t solverStat = cusolverDnCreate(&cusolverH);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("shapeExtract CUSOLVER initialization failed%d\n",solverStat);exit( -1 );}
    

            float *d_EigVal; 
            float *d_work;
            int *devInfo; 
	        int lwork = 0;

            CHECK(cudaMalloc((void**)&d_EigVal, sizeof(float)*MAT_COLS));
            CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	        solverStat = cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, MAT_COLS, 
                                                     d_result+classes_scan[i]*MAT_COLS*MAT_COLS, 
                                                     MAT_COLS, 
                                                     d_EigVal, &lwork);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set buffer failed%d\n",solverStat);exit( -1 );}
            
	        CHECK(cudaMalloc((void**)&d_work, sizeof(float)*lwork));


            solverStat = cusolverDnSsyevd(cusolverH, jobz, uplo, MAT_COLS, 
                                          d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS, 
                                          d_EigVal, d_work, lwork, devInfo);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER cusolverDnSsyevd failed%d\n",solverStat);exit( -1 );}
            
            cudaFree(d_EigVal);
            cudaFree(d_work);
            cudaFree(devInfo);
            cusolverDnDestroy(cusolverH);

            int c[1];
            c[0] = i;
            int *d_c;
            CHECK(cudaMalloc((void**)&d_c, sizeof(int)));
            cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);
            get_center<<<1, 1>>>(d_vals, d_result+classes_scan[i]*MAT_COLS*MAT_COLS+(MAT_COLS-1)*MAT_COLS,
                                 d_centers, d_scan_kcount, k, MAT_COLS, d_c);
            cudaDeviceSynchronize();
        }
        
    }

    cudaFree(d_iterv);
    cudaFree(d_result);
    for (int i = 0; i < k; i++)
    {
        cublasDestroy(handle[i]);
        cudaStreamDestroy(streams[i]);
    }

}




template <typename T>
typename std::enable_if<(sizeof(T) == 8)>::type get_class_centers_a(double *d_mat_seg, double *d_vals, 
                                                                  double *d_centers,
                                                                  int *d_scan_kcount, int *scan_kcount, 
                                                                  int *k_count, const int k, 
                                                                  const int MAT_COLS)
{
        
    double alpha = 1.0, beta = 0.0;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((MAT_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (MAT_COLS+threadsPerBlock.y-1)/threadsPerBlock.y);   

    T *d_matQ;
    CHECK(cudaMalloc((void**)&d_matQ, sizeof(T)*MAT_COLS*MAT_COLS));
    get_matQ<<<numBlocks, threadsPerBlock>>>(d_matQ, MAT_COLS, MAT_COLS);
    cudaDeviceSynchronize();

    double *d_result;
    CHECK(cudaMalloc((void**)&d_result, sizeof(double)*MAT_COLS*MAT_COLS)); 
    cudaMemset(d_result, 0, sizeof(double)*MAT_COLS*MAT_COLS);


    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            cublasHandle_t handle;
            cusolverDnHandle_t cusolverH;
            cublasStatus_t blasStat = cublasCreate(&handle);
            cusolverStatus_t solverStat = cusolverDnCreate(&cusolverH);

            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("shapeExtract CUBLAS initialization failed%d\n",blasStat);exit( -1 );}
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("shapeExtract CUSOLVER initialization failed%d\n",solverStat);exit( -1 );}
    
            blasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,   
                                   MAT_COLS, MAT_COLS, k_count[i],          
                                   &alpha, d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS,          
                                   d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS, &beta,             
                                   d_result, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}
            

            //Q*S
            blasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                                   MAT_COLS, MAT_COLS, MAT_COLS,          
                                   &alpha, d_result, MAT_COLS,          
                                   d_matQ, MAT_COLS, &beta,             
                                   d_result, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}

            //(Q*S)*Q
            blasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                        MAT_COLS, MAT_COLS, MAT_COLS,          
                        &alpha, d_matQ, MAT_COLS,          
                        d_result, MAT_COLS, &beta,             
                        d_result, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}

            double *d_EigVal; 
            double *d_work;
            int *devInfo; 
	        int lwork = 0;

            CHECK(cudaMalloc((void**)&d_EigVal, sizeof(double)*MAT_COLS));
            CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
            
	        solverStat = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, MAT_COLS, 
                                                     d_result, MAT_COLS, 
                                                     d_EigVal, &lwork);
            
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set buffer failed%d\n",solverStat);exit( -1 );}
            
	        CHECK(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

            
            solverStat = cusolverDnDsyevd(cusolverH, jobz, uplo, MAT_COLS, 
                                          d_result, MAT_COLS, 
                                          d_EigVal, d_work, lwork, devInfo);
            
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER cusolverDnSsyevd failed%d\n",solverStat);exit( -1 );}
            
            cudaFree(d_EigVal);
            cudaFree(d_work);
            cudaFree(devInfo);
            cublasDestroy(handle);
            cusolverDnDestroy(cusolverH);

            int c[1];
            c[0] = i;
            int *d_c;
            CHECK(cudaMalloc((void**)&d_c, sizeof(int)));
            cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);
            get_center<<<1, 1>>>(d_vals, d_result+(MAT_COLS-1)*MAT_COLS, d_centers, d_scan_kcount, k, MAT_COLS, d_c);
            cudaDeviceSynchronize();
            
        }
        
    }   
    cudaFree(d_matQ);
    cudaFree(d_result);

}






template <typename T>
typename std::enable_if<(sizeof(T) == 8)>::type get_class_centers(double *d_mat_seg, double *d_vals, 
                                                                  double *d_centers,
                                                                  int *d_scan_kcount, int *scan_kcount, 
                                                                  int *k_count, const int k, 
                                                                  const int MAT_COLS,
                                                                  int &sum_classes)
{
    cudaStream_t *streams = (cudaStream_t*)malloc(k*sizeof(cudaStream_t));
    cublasHandle_t *handle = (cublasHandle_t*)malloc(k*sizeof(cublasHandle_t));
    

    cublasStatus_t blasStat;
    cudaError_t streamStat;
    

    sum_classes = 0;
    int *classes_scan = (int*)malloc(k*sizeof(int));
    classes_scan[0] = 0;
    for (int i = 0; i < k-1; i++)
    {
        if(k_count[i] != 0)
            classes_scan[i+1] = classes_scan[i]+1;
        else
            classes_scan[i+1] = classes_scan[i];
    }
    
    for(int i=0; i<k; i++)
    {
         blasStat = cublasCreate(&handle[i]);
         if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("shapeExtract CUBLAS initialization failed%d\n",blasStat);exit( -1 );}
         streamStat = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
         if (streamStat != cudaSuccess) {printf ("shapeExtract cudaStream initialization failed%d\n",streamStat);exit( -1 );}
         if(k_count[i] != 0)
            sum_classes += 1;
    }

    //printf("sum_class: %d\n", sum_classes);

    double *d_result;
    CHECK(cudaMalloc((void**)&d_result, sizeof(double)*sum_classes*MAT_COLS*MAT_COLS)); 
    cudaMemset(d_result, 0, sizeof(double)*sum_classes*MAT_COLS*MAT_COLS);

        
    double alpha = 1.0, beta = 0.0;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((MAT_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (MAT_COLS+threadsPerBlock.y-1)/threadsPerBlock.y);   

    for(int i=0; i<k; i++)
    {
        blasStat = cublasSetStream(handle[i], streams[i]);
        if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS stream set failed%d\n",blasStat);exit( -1 );}

    }

    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            blasStat = cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_T,   
                                   MAT_COLS, MAT_COLS, k_count[i],          
                                   &alpha, d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS,          
                                   d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS, &beta,             
                                   d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}
        }
    }   
            
    double *d_matQ;
    CHECK(cudaMalloc((void**)&d_matQ, sizeof(double)*MAT_COLS*MAT_COLS));
    get_matQ<<<numBlocks, threadsPerBlock>>>(d_matQ, MAT_COLS, MAT_COLS);
    cudaDeviceSynchronize();
    
    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            //Q*S
            blasStat = cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                                   MAT_COLS, MAT_COLS, MAT_COLS,          
                                   &alpha, d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS,          
                                   d_matQ, MAT_COLS, &beta,             
                                   d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}
        }
    }

    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            //(Q*S)*Q
            blasStat = cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                        MAT_COLS, MAT_COLS, MAT_COLS,          
                        &alpha, d_matQ, MAT_COLS,          
                        d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS, &beta,             
                        d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}
        }
    }
    
    cudaFree(d_matQ);


    double *d_iterv;
    CHECK(cudaMalloc((void**)&d_iterv, sizeof(double)*sum_classes*MAT_COLS));  
    int initvecThreads = 512;
    int initvecBlocks = sum_classes;
    int initvectorBlocks = (MAT_COLS*sum_classes+initvecThreads-1)/initvecThreads;
    //one block -- one vector   
    init_vec<double><<<initvecBlocks, initvecThreads>>>(d_iterv, MAT_COLS);
    cudaDeviceSynchronize();

    const int normThreads = 1024;
    const int normBlocks = (MAT_COLS+normThreads-1)/normThreads;
    //const int uninormBlocks = sum_classes;
    
    for(int j = 0; j < 1; j++)
    {
        for (int i=0; i<k; i++)
        {
            if(k_count[i] != 0)
            {
                cublasDgemv(handle[i], CUBLAS_OP_T, MAT_COLS, MAT_COLS,
                        &alpha, d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS,
                        d_iterv+classes_scan[i]*MAT_COLS, 1, &beta, 
                        d_iterv+classes_scan[i]*MAT_COLS, 1);
                
                
                norm_vec<double><<<normBlocks, normThreads, 0, streams[i]>>>
                                    (d_iterv+classes_scan[i]*MAT_COLS, MAT_COLS);
                cudaDeviceSynchronize();
                
            }
        }
    }

    double *d_lamdavec;
    double *d_lamda;
    double *h_lamda = (double *) malloc(sizeof(double)*k);
    CHECK(cudaMalloc((void**)&d_lamdavec, sizeof(double)*sum_classes*MAT_COLS));
    CHECK(cudaMalloc((void**)&d_lamda, sizeof(double)*k));
    CHECK(cudaMemset(d_lamda, 0, sizeof(double)*k));
    CHECK(cudaMemcpy(d_lamdavec, d_iterv, sizeof(double)*sum_classes*MAT_COLS, cudaMemcpyDeviceToDevice));
    for(int i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            cublasDgemv(handle[i], CUBLAS_OP_T, MAT_COLS, MAT_COLS,
                        &alpha, d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS,
                        d_lamdavec+classes_scan[i]*MAT_COLS, 1, 
                        &beta, d_lamdavec+classes_scan[i]*MAT_COLS, 1);
            dot_reduce<double><<<1, normThreads, 0, streams[i]>>>
                    (d_iterv+classes_scan[i]*MAT_COLS, d_lamdavec+classes_scan[i]*MAT_COLS,
                     MAT_COLS, d_lamda+i);
            cudaDeviceSynchronize();
        }
    }
    CHECK(cudaMemcpy(h_lamda, d_lamda, sizeof(double)*sum_classes, cudaMemcpyDeviceToHost));
    cudaFree(d_lamdavec);
    cudaFree(d_lamda);
    

    

    for(int i=0; i<k; i++)
    {
        
        if(k_count[i] != 0 && h_lamda[i] > 0)
        {
            int c[1];
            c[0] = i;
            int *d_c;
            CHECK(cudaMalloc((void**)&d_c, sizeof(int)));
            cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);
            get_center<<<1, 1>>>(d_vals, d_iterv+classes_scan[i]*MAT_COLS, d_centers, 
                                 d_scan_kcount, k, MAT_COLS, d_c);
            cudaDeviceSynchronize();
        }
        if(k_count[i] != 0 && h_lamda[i] <= 0)
        {
            cusolverDnHandle_t cusolverH;
            cusolverStatus_t solverStat = cusolverDnCreate(&cusolverH);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("shapeExtract CUSOLVER initialization failed%d\n",solverStat);exit( -1 );}
    

            double *d_EigVal; 
            double *d_work;
            int *devInfo; 
	        int lwork = 0;

            CHECK(cudaMalloc((void**)&d_EigVal, sizeof(double)*MAT_COLS));
            CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	        solverStat = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, MAT_COLS, 
                                                     d_result+classes_scan[i]*MAT_COLS*MAT_COLS, 
                                                     MAT_COLS, 
                                                     d_EigVal, &lwork);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set buffer failed%d\n",solverStat);exit( -1 );}
            
	        CHECK(cudaMalloc((void**)&d_work, sizeof(double)*lwork));


            solverStat = cusolverDnDsyevd(cusolverH, jobz, uplo, MAT_COLS, 
                                          d_result+classes_scan[i]*MAT_COLS*MAT_COLS, MAT_COLS, 
                                          d_EigVal, d_work, lwork, devInfo);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER cusolverDnSsyevd failed%d\n",solverStat);exit( -1 );}
            
            cudaFree(d_EigVal);
            cudaFree(d_work);
            cudaFree(devInfo);
            cusolverDnDestroy(cusolverH);

            int c[1];
            c[0] = i;
            int *d_c;
            CHECK(cudaMalloc((void**)&d_c, sizeof(int)));
            cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyHostToDevice);
            get_center<<<1, 1>>>(d_vals, d_result+classes_scan[i]*MAT_COLS*MAT_COLS+(MAT_COLS-1)*MAT_COLS,
                                 d_centers, d_scan_kcount, k, MAT_COLS, d_c);
            cudaDeviceSynchronize();
        }
        
    }

    cudaFree(d_iterv);
    cudaFree(d_result);
    for (int i = 0; i < k; i++)
    {
        cublasDestroy(handle[i]);
        cudaStreamDestroy(streams[i]);
    }


}


template<typename T>
void extract_shape(int *d_idx, const T *d_mat, int *d_k_count, T *d_centers,
                   const int k, const int idx_len, const int mat_row,
                   const int mat_col, int &sum_class, const int cal_select)
{
    if(idx_len < 2e6)
    {
        class_count<int><<<(idx_len + 256 -1)/256, 256>>>(d_idx, idx_len, d_k_count);
        cudaDeviceSynchronize();
    }
    else
    {
        int *d_count_tmp;
        const int step_size = 1000;
        const int stride = (idx_len%step_size == 0) ? (idx_len/step_size) : (idx_len/step_size+1);
        CHECK(cudaMalloc((void**)&d_count_tmp, sizeof(int)*stride*k));
        histo_split<<<idx_len/step_size/128+1, 128>>>(d_idx, d_count_tmp, step_size, k, idx_len);
        cudaDeviceSynchronize();
        histo_merge<<<idx_len/step_size/128+1, 128>>>(d_idx, d_count_tmp, d_k_count, step_size, k, idx_len); 
        cudaDeviceSynchronize();
        cudaFree(d_count_tmp);
    }

    int *d_loc2;
    int *count_k_cur;
    int *d_scan_kcount;
    CHECK(cudaMalloc((void**)&d_loc2, sizeof(int)*idx_len));
    CHECK(cudaMalloc((void**)&count_k_cur, sizeof(int)*k));
    CHECK(cudaMalloc((void**)&d_scan_kcount, sizeof(int)*k));
    cudaMemset(count_k_cur, 0, sizeof(int)*k);

    if(mat_row < 1e6)
    {
        get_loc2<<<1, 1>>>(d_idx, count_k_cur, d_loc2, idx_len);
        cudaDeviceSynchronize();
    }
    else
    {
        int *d_count_tmp;
        const int step_size = 1000;
        const int stride = (mat_row%step_size == 0) ? (mat_row/step_size) : (mat_row/step_size+1);
        CHECK(cudaMalloc((void**)&d_count_tmp, sizeof(int)*stride*k));
        loc2_split<<<mat_row/step_size/128+1, 128>>>(d_idx, d_count_tmp, d_loc2, step_size, k, mat_row);
        cudaDeviceSynchronize();
        per_scan<<<(k+32-1)/32, 32>>>(d_count_tmp, step_size, k, mat_row);
        cudaDeviceSynchronize();
        loc2_merge<<<mat_row/step_size/128+1, 128>>>(d_idx, d_count_tmp, d_loc2, step_size, k, mat_row);
    
        cudaDeviceSynchronize();
        cudaFree(d_count_tmp);
    }

    
    thrust::exclusive_scan(thrust::device, d_k_count, d_k_count + k, d_scan_kcount, 0);
    cudaDeviceSynchronize();
    

    T *d_mat_seg;
    CHECK(cudaMalloc((void**)&d_mat_seg, sizeof(T)*mat_row*mat_col));
   
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((mat_row+threadsPerBlock.x-1)/threadsPerBlock.x, 
                   (mat_col+threadsPerBlock.y-1)/threadsPerBlock.y);
    block_alloc2D<<<numBlocks, threadsPerBlock>>>(d_mat_seg, d_mat, d_idx, d_scan_kcount,
                                                  d_loc2, mat_row, mat_col);
    cudaDeviceSynchronize();

    int *k_count = (int*)malloc(sizeof(int)*k); 
    int *scan_kcount = (int*)malloc(sizeof(int)*k);
    cudaMemcpy(k_count, d_k_count, sizeof(int)*k, cudaMemcpyDeviceToHost);
    cudaMemcpy(scan_kcount, d_scan_kcount, sizeof(int)*k, cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; i++)
    {
        T center_sum = thrust::reduce(thrust::device, d_centers+mat_col*i, d_centers+mat_col*(i+1), 0);
        cudaDeviceSynchronize();
        center_sum = center_sum>0 ? center_sum : (-1)*center_sum;
        int cur_count = k_count[i];
        if(center_sum > 1e-16 && cur_count != 0)
        {
            sbd3D(d_mat_seg+scan_kcount[i]*mat_col, d_centers+mat_col*i, d_mat_seg+scan_kcount[i]*mat_col, cur_count, mat_col);
        }
    }

    z_norm_gpu_x(d_mat_seg, mat_row, mat_col, 1);

    T *d_mat_val;
    CHECK(cudaMalloc((void**)&d_mat_val, sizeof(T)*k*mat_col));
    val_copy<<<(k+256-1)/256, 256>>>(d_mat_seg, d_mat_val, d_scan_kcount, k, mat_col);
    cudaDeviceSynchronize();


    if(sizeof(T) == sizeof(double))
    {
        get_class_centers_a<T>(d_mat_seg, d_mat_val, d_centers, 
                         d_scan_kcount, scan_kcount, k_count, 
                         k, mat_col);
    }
    else
    {
        if(cal_select == 0)
        {
            get_class_centers<T>(d_mat_seg, d_mat_val, d_centers, 
                         d_scan_kcount, scan_kcount, k_count, 
                         k, mat_col, sum_class);
        }
    
        else
        {
            get_class_centers_a<T>(d_mat_seg, d_mat_val, d_centers, 
                         d_scan_kcount, scan_kcount, k_count, 
                         k, mat_col);
        }        
    }
        
    
    free(k_count);
    free(scan_kcount);
    cudaFree(d_mat_val);
    cudaFree(d_mat_seg);
    cudaFree(d_scan_kcount);
    cudaFree(d_loc2);
    cudaFree(count_k_cur);
    
}


#endif