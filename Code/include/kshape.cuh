#ifndef GPU_KSHAPE__
#define GPU_KSHAPE__

#include "cuda_runtime.h"
#include <cstdlib>
#include <cuda.h>
#include <random>
#include <set>
#include <curand.h>
#include <curand_kernel.h>
#include "shapeExtract.cuh"
#include "timer.cuh"
#include "check.cuh"
#include "print_test.cuh"

template<typename T>
__device__ void rand_init(unsigned int seed, T &result, const int k) 
{
    curandState_t state;
    curand_init(seed, 0, 0, &state);
    result = curand(&state) % k + 1;
}

template<typename T>
__global__ void idx_init(T *d_idx, const int k, const int mat_row)
{
    const unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < mat_row)
    {
        rand_init(idx, d_idx[idx], k); 
    }
}


template<typename T>
__global__ void get_pos_idx(T *d_mat_ncc, int *d_idx, T *part_sum,
                            const int k, const int ncc_len, const int mat_row)
{
    const int idx = threadIdx.x+blockDim.x*blockIdx.x;
    const int tid = threadIdx.x;
    const int bsize = blockDim.x;
    extern __shared__ T sh_arr[];
    if (idx < mat_row*bsize)
    {
        int tmp = idx/bsize;
        __shared__ T min_val;
        min_val = 1e10;
        int pos = 0;
        for (int i = 0; i < k; i++)
        {
            int off = tmp*k*ncc_len+i*ncc_len;
            if(tid < ncc_len)
            {
                sh_arr[tid] = (1.0-d_mat_ncc[off+tid]);
            }
            else
            {
                 //sh_arr[tid] = 0; 
                 sh_arr[tid] = 2; 
            }
            T cmp_tmp;
            for (int j = tid+bsize; j < ncc_len; j+=bsize)
            {
                //sh_arr[tid] += (1.0-d_mat_ncc[off+j]);
                cmp_tmp = (1.0-d_mat_ncc[off+j]);
                sh_arr[tid] = sh_arr[tid]>cmp_tmp?cmp_tmp:sh_arr[tid];
            }
            __syncthreads();
            for (int activeTrheads=bsize/2; activeTrheads>0; activeTrheads >>= 1)
            {
                if (tid < activeTrheads)
                {
                    //sh_arr[tid] += sh_arr[tid + activeTrheads];
                    cmp_tmp = sh_arr[tid + activeTrheads];
                    sh_arr[tid] = sh_arr[tid]>cmp_tmp?cmp_tmp:sh_arr[tid];
                }
                __syncthreads();
            }
            if(tid == 0)
            {
                //sh_arr[0] /= ncc_len;
                if(sh_arr[0]<min_val)
                {
                    min_val = sh_arr[0];
                    pos = i;
                }
            }    
        }
        if(tid == 0)
        {
            d_idx[idx/bsize] = pos+1; 
            part_sum[idx/bsize] = min_val;
        } 
    }
    
}

/*
template<typename T>
__global__ void center_distance(T *d_mat, T *d_centers, int *d_idx, T *part_sum,
                                const int mat_row, const int mat_col)
{
    const int idx = threadIdx.x+blockDim.x*blockIdx.x;
    const int tid = threadIdx.x;
    const int bsize = blockDim.x;
    extern __shared__ T sh_arr[];
    if (idx < mat_row*bsize)
    {
        int tmp = idx/bsize;
        int off = tmp*mat_col;
        int off_center = (d_idx[tmp]-1)*mat_col;
        T mut_tmp;
        if(tid < mat_col)
        {
            mut_tmp = d_centers[off_center+tid]-d_mat[off+tid];
            sh_arr[tid] = mut_tmp*mut_tmp;
        }
        else
        {
            sh_arr[tid] = 0; 
        }
        T cmp_tmp;
        for (int j = tid+bsize; j < mat_col; j+=bsize)
        {
            mut_tmp = d_centers[off_center+j]-d_mat[off+j];
            sh_arr[tid] += (mut_tmp*mut_tmp);
        }
        __syncthreads();
        for (int activeTrheads=bsize/2; activeTrheads>0; activeTrheads >>= 1)
        {
            if (tid < activeTrheads)
            {
                sh_arr[tid] += sh_arr[tid + activeTrheads];
            }
            __syncthreads();
        }
        if(tid == 0)
        {
            part_sum[idx/bsize] = sh_arr[0]/mat_col;
        } 
    }
}
*/

template<typename T>
__global__ void is_equal(T *d_old_idx, T *d_idx, const int mat_row, T *flag)
{
    const int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < mat_row)
    {
        if(d_old_idx[idx] != d_idx[idx])
             flag[0] = 0;
    }
}




vector<int> randomSelectK(const int n, const int k) {
    vector<int> result;
    set<int> selected;
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n-1);
    
    for (int i = 0; i < k; i++) {
        int num;
        do {
            num = dis(gen);
        } while (selected.find(num) != selected.end());
        selected.insert(num);
        result.push_back(num);
    }
    return result;
}

template<typename T>
T * center_select(vector<int> ranSeltId, const T *h_mat, const int mat_row, const int mat_col)
{
    int len = ranSeltId.size();
    T *center = new T[len*mat_col];
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < mat_col; j++)
        {
            center[i*mat_col+j] = h_mat[ranSeltId[i]*mat_col+j];
        }
    }
    return center;
}





template<typename T>
void k_shape(const T *d_mat, int *idx,
             const int mat_row, const int mat_col, const int k,
             int &iter_num, int &sum_class, const int cal_select)
{
    int *d_idx;
    int *d_old_idx;
    T *d_centers;
    int *d_kcount;
    T *ncc_out;
    T *part_sum;
    const int ncc_len = 2*mat_col-1;
    CHECK(cudaMalloc((void**)&d_idx, sizeof(int)*mat_row));
    CHECK(cudaMalloc((void**)&d_old_idx, sizeof(int)*mat_row));
    CHECK(cudaMalloc((void**)&d_centers, sizeof(T)*k*mat_col));
    cudaMemset(d_centers, 0, sizeof(T)*k*mat_col);
    CHECK(cudaMalloc((void**)&d_kcount, sizeof(int)*k));
    CHECK(cudaMalloc((void**)&ncc_out, sizeof(T)*mat_row*k*ncc_len));
    CHECK(cudaMalloc((void**)&part_sum, sizeof(T)*mat_row));
    cudaMemset(ncc_out, 0, sizeof(T)*mat_row*k*ncc_len);
    const int blockSize = 32;
    const int gridSize = (mat_row+blockSize-1)/blockSize;
    const int gridSizePos = mat_row;
    const int blockSizePos = 512;
    
    idx_init<<<gridSize, blockSize>>>(d_idx, k, mat_row);
    cudaDeviceSynchronize();
    
    cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);

    int *d_flag;
    CHECK(cudaMalloc((void**)&d_flag, sizeof(int)));
    
    std::cout<<"start iter"<<std::endl;
    GpuTimer timer;
    int flag[1];
    double old_inertia = 1e10;
    int countIter = 0;
    for (int i = 0; i < iter_num; i++)
    {
        std::cout<<"iter ------------------->i: "<<i<<std::endl;

        flag[0] = 1;
        cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_kcount, 0, sizeof(int)*k);       
        
        std::cout<<"start calcute extract_shape"<<std::endl;
        timer.Start();
        extract_shape(d_idx, d_mat, d_kcount, d_centers, k, mat_row, mat_row, mat_col, sum_class, cal_select);
        timer.Stop();

        if(sum_class == 1 || sum_class<=(k/2))
            break;
        
        printf("extract_shape run on GPU: %f msecs.\n", timer.Elapsed());
        //err in centers
        //print_test_f<<<1,1>>>(d_centers, k, mat_col);
        
        std::cout<<"start calcute NCC_3D"<<std::endl;
        timer.Start();
        NCC_3D(d_mat, d_centers, ncc_out, mat_row, k, mat_col);
        timer.Stop();
        printf("NCC_3D run on GPU: %f msecs.\n", timer.Elapsed());
        
        //一个元素内ncc_len步寻找最大值及其对应的索引，k个值之间进行比较
        //一步并行度 mat_row,二步并行度ncc_len,串行度k 
        std::cout<<"start calcute get_pos_idx"<<std::endl;
        timer.Start();
        get_pos_idx<<<gridSizePos, blockSizePos, sizeof(T)*blockSizePos>>>(ncc_out, d_idx, part_sum, k, ncc_len, mat_row);
        cudaDeviceSynchronize(); 
        
        //center_distance<<<gridSizePos, blockSizePos, sizeof(T)*blockSizePos>>>(d_mat, d_centers, d_idx, part_sum, mat_row, mat_col);
        //cudaDeviceSynchronize();
        double inertia = double(thrust::reduce(thrust::device, part_sum, part_sum+mat_row, 0.0));
        inertia /= mat_row;
        timer.Stop();
        printf("get_pos_idx run on GPU: %f msecs.\n", timer.Elapsed());  

        std::cout<<"start calcute is_equal"<<std::endl;
        timer.Start();
        is_equal<<<gridSize, blockSize>>>(d_old_idx, d_idx, mat_row, d_flag);
        cudaDeviceSynchronize();
        cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        timer.Stop();
        printf("is_equal run on GPU: %f msecs.\n", timer.Elapsed());

        /*
        if (flag[0])
        {
            break;
        }  
        cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);
        old_inertia = inertia;
        */
        //countIter = 0;

        //if (countIter>10)
        
        if (flag[0] || countIter>20)
        {
            iter_num = i+1;
            break;
        }  
        else if(inertia<old_inertia || abs(old_inertia-inertia)<1e-6)
        {
            cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);
            old_inertia = inertia;
            countIter = 0;
        }   
        else
            countIter++;
        
        printf("inertia is: %lf \n", old_inertia);
        std::cout<<std::endl;
    }

    std::cout<<"iter end"<<std::endl;
    
    cudaMemcpy(idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToHost);

    cudaFree(d_flag);
    cudaFree(d_old_idx);
    cudaFree(ncc_out);
    cudaFree(d_idx);
    cudaFree(d_centers);
    cudaFree(d_kcount);
    cudaFree(part_sum);
}







template<typename T>
void k_shape_tsl(const T *d_mat, const T *h_mat, int *idx,
             const int mat_row, const int mat_col, const int k,
             int &iter_num, int &sum_class, const int cal_select)
{
    int *d_idx;
    int *d_old_idx;
    T *d_centers;
    int *d_kcount;
    T *ncc_out;
    T *part_sum;
    const int ncc_len = 2*mat_col-1;

    vector<int> randidx = randomSelectK(mat_row, k);
    T *h_init_center = center_select(randidx, h_mat, mat_row, mat_col);

    CHECK(cudaMalloc((void**)&d_idx, sizeof(int)*mat_row));
    CHECK(cudaMalloc((void**)&d_old_idx, sizeof(int)*mat_row));
    CHECK(cudaMalloc((void**)&d_centers, sizeof(T)*k*mat_col));
    cudaMemset(d_centers, 0, sizeof(T)*k*mat_col);
    cudaMemcpy(d_centers, h_init_center, sizeof(T)*k*mat_col, cudaMemcpyHostToDevice);

    CHECK(cudaMalloc((void**)&d_kcount, sizeof(int)*k));
    CHECK(cudaMalloc((void**)&ncc_out, sizeof(T)*mat_row*k*ncc_len));
    CHECK(cudaMalloc((void**)&part_sum, sizeof(T)*mat_row));
    cudaMemset(ncc_out, 0, sizeof(T)*mat_row*k*ncc_len);
    const int blockSize = 32;
    const int gridSize = (mat_row+blockSize-1)/blockSize;
    const int gridSizePos = mat_row;
    const int blockSizePos = 512;
    
    idx_init<<<gridSize, blockSize>>>(d_idx, k, mat_row);
    cudaDeviceSynchronize();
    
    cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);

    int *d_flag;
    CHECK(cudaMalloc((void**)&d_flag, sizeof(int)));
    
    std::cout<<"start iter"<<std::endl;
    GpuTimer timer;
    int flag[1];
    double old_inertia = 1e10;
    int countIter = 0;
    for (int i = 0; i < iter_num; i++)
    {
        std::cout<<"iter ------------------->i: "<<i<<std::endl;

        flag[0] = 1;
        cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);
        
        //err in centers
        //print_test_f<<<1,1>>>(d_centers, k, mat_col);
        
        //NCC compute ....... 
        std::cout<<"start calcute NCC_3D"<<std::endl;
        timer.Start();
        NCC_3D(d_mat, d_centers, ncc_out, mat_row, k, mat_col);
        timer.Stop();
        printf("NCC_3D run on GPU: %f msecs.\n", timer.Elapsed());
        
        //一个元素内ncc_len步寻找最大值及其对应的索引，k个值之间进行比较
        //一步并行度 mat_row,二步并行度ncc_len,串行度k 
        std::cout<<"start calcute get_pos_idx"<<std::endl;
        timer.Start();
        get_pos_idx<<<gridSizePos, blockSizePos, sizeof(T)*blockSizePos>>>(ncc_out, d_idx, part_sum, k, ncc_len, mat_row);
        cudaDeviceSynchronize(); 
        
        //center_distance<<<gridSizePos, blockSizePos, sizeof(T)*blockSizePos>>>(d_mat, d_centers, d_idx, part_sum, mat_row, mat_col);
        //cudaDeviceSynchronize();
        double inertia = double(thrust::reduce(thrust::device, part_sum, part_sum+mat_row, 0.0));
        inertia /= mat_row;
        timer.Stop();
        printf("get_pos_idx run on GPU: %f msecs.\n", timer.Elapsed());  


        //max reduce And judge ........
        std::cout<<"start calcute is_equal"<<std::endl;
        timer.Start();
        is_equal<<<gridSize, blockSize>>>(d_old_idx, d_idx, mat_row, d_flag);
        cudaDeviceSynchronize();
        cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        timer.Stop();
        printf("is_equal run on GPU: %f msecs.\n", timer.Elapsed());

        /*
        if (flag[0])
        {
            break;
        }  
        cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);
        old_inertia = inertia;
        */
        //countIter = 0;

        //if (countIter>10)
        
        if (flag[0] || countIter>20)
        {
            iter_num = i+1;
            break;
        }  
        else if(inertia<old_inertia || abs(old_inertia-inertia)<1e-6)
        {
            cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);
            old_inertia = inertia;
            countIter = 0;
        }   
        else
            countIter++;
        
        printf("inertia is: %lf \n", old_inertia);


        //shape extract.........
        cudaMemset(d_kcount, 0, sizeof(int)*k);       
        
        std::cout<<"start calcute extract_shape"<<std::endl;
        timer.Start();
        extract_shape(d_idx, d_mat, d_kcount, d_centers, k, mat_row, mat_row, mat_col, sum_class, cal_select);
        timer.Stop();

        if(sum_class == 1 || sum_class<=(k/2))
            break;
        
        printf("extract_shape run on GPU: %f msecs.\n", timer.Elapsed());


        std::cout<<std::endl;
    }

    std::cout<<"iter end"<<std::endl;
    
    cudaMemcpy(idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToHost);

    cudaFree(d_flag);
    cudaFree(d_old_idx);
    cudaFree(ncc_out);
    cudaFree(d_idx);
    cudaFree(d_centers);
    cudaFree(d_kcount);
    cudaFree(part_sum);
}





#endif