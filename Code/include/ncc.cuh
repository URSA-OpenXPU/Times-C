#ifndef GPU_NCC__
#define GPU_NCC__

#include <cuda_runtime.h>
#include <cufft.h>  
#include <cuComplex.h>
#include "check.cuh"




template<typename T>
static inline int my_log2(T x)
{
    return log(x)/log(2);
}


template<typename T>
__device__ void sum_square(T *d_mat, const int mat_col, const int cur, T &sum)
{
    for (int idx = 0; idx < mat_col; idx++)
    {
        sum += d_mat[idx + cur*mat_col] * d_mat[idx + cur*mat_col];
    }
    sum = sqrtf(sum);
}


template<typename T>
__global__ void Mat_norm(T *d_mat_x, T *d_mat_y, T *d_mat_out, 
                         const int mat_x_row, const int mat_y_row,
                         const int mat_col)
{
    const int max_row = mat_x_row>mat_y_row?mat_x_row:mat_y_row;
    const int min_row = mat_x_row<mat_y_row?mat_x_row:mat_y_row;
    bool flag = mat_x_row>mat_y_row?true:false;
    
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < max_row)
    {
        if(flag)
        {
            T init = 0;
            T sum_x = 0;
            sum_square(d_mat_x, mat_col, idx, sum_x);
            d_mat_out[idx] = sum_x;
            if(idx < min_row)
            {
                T sum_y = 0;
                sum_square(d_mat_y, mat_col, idx, sum_y);
                d_mat_out[mat_x_row + idx] = sum_y;
            }
        }
        else
        {
            T sum_y = 0;
            sum_square(d_mat_y, mat_col, idx, sum_y);
            
            d_mat_out[mat_x_row + idx] = sum_y;
            if(idx < min_row)
            {
                T sum_x = 0;
                sum_square(d_mat_x, mat_col, idx, sum_x);
                
                d_mat_out[idx] = sum_x;
            }
        }
        
    }
}



template<typename T>
__global__ void Mat_norm_dot(T *d_mat_tmp, T *d_mat_out, 
                             const int mat_x_row, const int mat_y_row,
                             bool flag)
{
    const int max_row = mat_x_row>mat_y_row?mat_x_row:mat_y_row;
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < max_row)
    {
        if(flag)
        {
            for (int i = 0; i < mat_y_row; i++)
            {
                d_mat_out[idx*mat_y_row + i] = d_mat_tmp[idx] * d_mat_tmp[mat_x_row+i];
                d_mat_out[idx*mat_y_row + i] = d_mat_out[idx*mat_y_row + i] == 0 ? 1e10:d_mat_out[idx*mat_y_row + i];
            }
        }
        else
        {
            for (int i = 0; i < mat_x_row; i++)
            {
                d_mat_out[idx+mat_y_row*i] = d_mat_tmp[i] * d_mat_tmp[mat_x_row+idx];
                d_mat_out[idx+mat_y_row*i] = d_mat_out[idx+mat_y_row*i] == 0 ? 1e10:d_mat_out[idx+mat_y_row*i];
            }
        }   
    }
}

template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type 
__device__ static __inline__ complex_dot(cuFloatComplex x, cuFloatComplex y, 
                                         cuFloatComplex &result, const float scale)
{
    result = cuCmulf(x, cuConjf(y));
    result = make_cuFloatComplex(scale*cuCrealf(result), scale*cuCimagf(result));
}

template <typename T>
typename std::enable_if<(sizeof(T) == 8)>::type 
__device__ static __inline__ complex_dot(cuDoubleComplex x, cuDoubleComplex y, 
                                         cuDoubleComplex &result, const double scale)
{
    result = cuCmul(x, cuConj(y));
    result = make_cuDoubleComplex(scale*cuCreal(result), scale*cuCimag(result));
}



template <typename T1, typename T2>
__global__ void complex_dot3D(T1 *d_mat_x, T1 *d_mat_y, T1 *d_out, T2 *sum_dot,
                          const int mat_x_row, const int mat_y_row, 
                          const int mat_fft_col)
{

    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int idy = threadIdx.y + blockDim.y*blockIdx.y;
    const int idz = threadIdx.z + blockDim.z*blockIdx.z;

    if(idx < mat_x_row && idy < mat_y_row && idz< mat_fft_col)
    {

        T2 dot = sum_dot[idy + idx*mat_y_row];
        T2 scale = 1/(mat_fft_col*dot);

        const int out_i = idz + (idy + idx*mat_y_row)*mat_fft_col;
        const int x_i = idz + idx*mat_fft_col;
        const int y_i = idz + idy*mat_fft_col;
        
        complex_dot<T2>(d_mat_x[x_i], d_mat_y[y_i], d_out[out_i], scale);
    }
}



template <typename T1, typename T2>
__global__ void complex_dot3D_unroll(T1 *d_mat_x, T1 *d_mat_y, T1 *d_out, T2 *sum_dot,
                          const int mat_x_row, const int mat_y_row, 
                          const int mat_fft_col)
{

    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    const int idy = threadIdx.y + blockDim.y*blockIdx.y;
    const int idz = threadIdx.z + blockDim.z*blockIdx.z;
    const int z_id = threadIdx.z;
    const int x_id = threadIdx.x;
    const int unroll_size = 4;

    const int z_gsize = (mat_fft_col+unroll_size-1)/unroll_size;
    
    extern __shared__ T1 sh_center[];

    if(idx < mat_x_row && idy < mat_y_row && idz*unroll_size < mat_fft_col && (idz+1)*unroll_size < mat_fft_col)
    {

        T2 dot = sum_dot[idy + idx*mat_y_row];
        T2 scale = 1/(mat_fft_col*dot);

        const int add_var = idz*unroll_size;
        const int out_i = add_var + (idy + idx*mat_y_row)*mat_fft_col;
        const int x_i = add_var + idx*mat_fft_col;
        const int y_i = add_var + idy*mat_fft_col;

        if(x_id == 0)
        {
            for (int i = 0; i < unroll_size; i++)
            {
                sh_center[z_id*unroll_size+i] = d_mat_y[y_i+i];
            }
            
        }
        __syncthreads();
        
        //4x unrolling
        complex_dot<T2>(d_mat_x[x_i], sh_center[z_id*unroll_size], d_out[out_i], scale);
        complex_dot<T2>(d_mat_x[x_i+1], sh_center[z_id*unroll_size+1], d_out[out_i+1], scale);
        complex_dot<T2>(d_mat_x[x_i+2], sh_center[z_id*unroll_size+2], d_out[out_i+2], scale);
        complex_dot<T2>(d_mat_x[x_i+3], sh_center[z_id*unroll_size+3], d_out[out_i+3], scale);
        

    }

    if(idx < mat_x_row && idy < mat_y_row && idz*unroll_size < mat_fft_col && (idz+1)*unroll_size >= mat_fft_col)
    {
        T2 dot = sum_dot[idy + idx*mat_y_row];
        T2 scale = 1/(mat_fft_col*dot);

        int iterNum = mat_fft_col - idz*unroll_size;
        const int add_var = idz*unroll_size;
        const int out_i = add_var + (idy + idx*mat_y_row)*mat_fft_col;
        const int x_i = add_var + idx*mat_fft_col;
        const int y_i = add_var + idy*mat_fft_col;

        if(x_id == 0)
        {
            for (int i = 0; i < iterNum; i++)
            {
                sh_center[z_id*unroll_size+i] = d_mat_y[y_i+i];
            }
        }
        __syncthreads();
        
        for (int i = 0; i < iterNum; i++)
        {
            complex_dot<T2>(d_mat_x[x_i+i], sh_center[z_id*unroll_size+i], d_out[out_i+i], scale);
        }
    }
}







template<typename T>
__global__ void copy_result2D(T *src, T *dest, const int src_col, 
                              const int dest_col, const int rows)
{
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const int idy = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(idx < rows && idy < dest_col)
    {
        const int pos_src = idx * src_col;
        const int pos_dest = idx * dest_col;
        
        const int offset = (idy+src_col-(dest_col+1)/2+1)%src_col;
        dest[pos_dest + idy] = src[pos_src + offset];
    }
    
}


template<typename T>
__global__ void copyPaddingData2D(const T *src, T *dest, const int rows,
                                  const int src_cols, const int dest_cols)
{
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const int idy = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(idx < rows && idy < src_cols)
    {
        dest[idx*dest_cols+idy] = src[idx*src_cols+idy];
    }
}


template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type excu_fft3D(const float *d_mat_x, const float *d_mat_y, 
                                                           float *fft_out, float *sum_dot,
                                                           const int mat_x_row,
                                                           const int mat_y_row, 
                                                           const int mat_col)
{
    const int fft_size = 1 << (int)ceil(my_log2((2*mat_col - 1)));

    int RANK = 1;
    int istride = 1;
    int idist = fft_size;
    int idist_out = fft_size;
	int ostride=1;
	int odist = fft_size;
    int odist_out = fft_size;
    int n_x[1],n_y[1],n_out[1];
    n_x[0] = fft_size;
    n_y[0] = fft_size;
    n_out[0] = fft_size;
    int inembed_x[2], inembed_y[2], inembed_out[3];
	int onembed_x[2], onembed_y[2], onembed_out[3];
	inembed_x[0]=mat_x_row;  onembed_x[0]=mat_x_row;
	inembed_x[1] = fft_size; onembed_x[1] = fft_size;
    inembed_y[0]=mat_y_row;  onembed_y[0]=mat_y_row;
	inembed_y[1] = fft_size; onembed_y[1] = fft_size;
    inembed_out[0] = mat_x_row; onembed_out[0] = mat_x_row;
    inembed_out[1] = mat_y_row; onembed_out[1] = mat_y_row;
    inembed_out[2] = fft_size; onembed_out[2] = fft_size;


    float *d_padding_x, *d_padding_y;
    cufftComplex *o_xdata, *o_ydata, *o_data;

    CHECK(cudaMalloc((void**)&d_padding_x, sizeof(float)*fft_size*mat_x_row));
    cudaMemset(d_padding_x, 0, sizeof(float)*fft_size*mat_x_row);
    CHECK(cudaMalloc((void**)&d_padding_y, sizeof(float)*fft_size*mat_y_row));
    cudaMemset(d_padding_y, 0, sizeof(float)*fft_size*mat_y_row);

    dim3 copyThreads(32, 32);
    dim3 copyBlocks_X((mat_x_row+copyThreads.x-1)/copyThreads.x, (mat_col+copyThreads.y-1)/copyThreads.y);
    dim3 copyBlocks_Y((mat_y_row+copyThreads.x-1)/copyThreads.x, (mat_col+copyThreads.y-1)/copyThreads.y);
    copyPaddingData2D<<<copyBlocks_X, copyThreads>>>(d_mat_x, d_padding_x, mat_x_row,
                                                     mat_col, fft_size);
    copyPaddingData2D<<<copyBlocks_Y, copyThreads>>>(d_mat_y, d_padding_y, mat_y_row,
                                                     mat_col, fft_size);
    cudaDeviceSynchronize();

    CHECK(cudaMalloc((void**)&o_xdata, sizeof(cufftComplex)*fft_size*mat_x_row));
    CHECK(cudaMalloc((void**)&o_ydata, sizeof(cufftComplex)*fft_size*mat_y_row));
    CHECK(cudaMalloc((void**)&o_data, sizeof(cufftComplex)*fft_size*mat_x_row*mat_y_row));

    cufftHandle fftPlanFwd_x;
    cufftHandle fftPlanFwd_y;
    cufftHandle fftPlanInv;
    cufftResult stat;

    stat = cufftPlanMany(&fftPlanFwd_x, RANK, n_x, inembed_x, 
                        istride, idist, onembed_x, 
                        ostride, odist, CUFFT_R2C, mat_x_row);
    if (stat != CUFFT_SUCCESS) {printf("cufft MakePlan error in fftPlanFwd_x %d\n",stat); exit( -1 );}
    stat = cufftPlanMany(&fftPlanFwd_y, RANK, n_y, inembed_y, 
                        istride, idist, onembed_y, 
                        ostride, odist, CUFFT_R2C, mat_y_row);
    if (stat != CUFFT_SUCCESS) {printf("cufft MakePlan error in fftPlanFwd_y %d\n",stat); exit( -1 );}
    stat = cufftPlanMany(&fftPlanInv, RANK, n_out, inembed_out, 
                        istride, idist_out, onembed_out, 
                        ostride, odist_out, CUFFT_C2R, mat_x_row*mat_y_row);
    if (stat != CUFFT_SUCCESS) {printf("cufft MakePlan error in fftPlanInv%d\n",stat); exit( -1 );}
    stat = cufftExecR2C(fftPlanFwd_x, d_padding_x, o_xdata);
    if (stat != CUFFT_SUCCESS) {printf("cufft cufftExecR2C error %d\n",stat); exit( -1 );}
    stat = cufftExecR2C(fftPlanFwd_y, d_padding_y, o_ydata);
    if (stat != CUFFT_SUCCESS) {printf("cufft cufftExecR2C error %d\n",stat); exit( -1 );}

    
    dim3 complexThreads(32, 2, 16);
    dim3 complexBlocks((mat_x_row+complexThreads.x-1)/complexThreads.x, 
                       (mat_y_row+complexThreads.y-1)/complexThreads.y,
                       (fft_size+complexThreads.z-1)/complexThreads.z);
    complex_dot3D<cufftComplex, float><<<complexBlocks, complexThreads>>>(o_xdata, o_ydata, o_data, sum_dot,
                                                     mat_x_row, mat_y_row, fft_size);
    
    
    /*
    dim3 complexThreads(32, 1, 16);
    const int unroll_size = 4;
    dim3 complexBlocks_unroll((mat_x_row+complexThreads.x-1)/complexThreads.x, 
                       (mat_y_row+complexThreads.y-1)/complexThreads.y,
                       ((fft_size+unroll_size-1)/unroll_size+complexThreads.z-1)/complexThreads.z);
    complex_dot3D_unroll<cufftComplex, float><<<complexBlocks_unroll, complexThreads, sizeof(cufftComplex)*complexThreads.z*unroll_size>>>
                        (o_xdata, o_ydata, o_data, sum_dot, mat_x_row, mat_y_row, fft_size);
    */
    
    cudaDeviceSynchronize();


    stat = cufftExecC2R(fftPlanInv, o_data, fft_out);
    if (stat != CUFFT_SUCCESS) {printf("cufft cufftExecC2R error %d\n",stat); exit( -1 );}

    cufftDestroy(fftPlanFwd_x);
    cufftDestroy(fftPlanFwd_y);
    cufftDestroy(fftPlanInv);
    cudaFree(d_padding_x);
    cudaFree(d_padding_y);
    cudaFree(o_xdata);
    cudaFree(o_ydata);
    cudaFree(o_data);
}

template <typename T>
typename std::enable_if<(sizeof(T) == 8)>::type excu_fft3D(const double *d_mat_x, const double *d_mat_y, 
                                                           double *fft_out, double *sum_dot,
                                                           const int mat_x_row,
                                                           const int mat_y_row, 
                                                           const int mat_col)
{
    const int fft_size = 1 << (int)ceil(my_log2((2*mat_col - 1)));

    int RANK = 1;
    int istride = 1;
    int idist = fft_size;
    int idist_out = fft_size;
	int ostride=1;
	int odist = fft_size;
    int odist_out = fft_size;
    int n_x[1],n_y[1],n_out[1];
    n_x[0] = fft_size;
    n_y[0] = fft_size;
    n_out[0] = fft_size;
    int inembed_x[2], inembed_y[2], inembed_out[3];
	int onembed_x[2], onembed_y[2], onembed_out[3];
	inembed_x[0]=mat_x_row;  onembed_x[0]=mat_x_row;
	inembed_x[1] = fft_size; onembed_x[0] = fft_size;
    inembed_y[0]=mat_y_row;  onembed_y[0]=mat_y_row;
	inembed_y[1] = fft_size; onembed_y[0] = fft_size;
    inembed_out[0] = mat_x_row; onembed_out[0] = mat_x_row;
    inembed_out[1] = mat_y_row; onembed_out[1] = mat_y_row;
    inembed_out[2] = fft_size; onembed_out[2] = fft_size;


    double *d_padding_x, *d_padding_y;
    cufftDoubleComplex *o_xdata, *o_ydata, *o_data;

    CHECK(cudaMalloc((void**)&d_padding_x, sizeof(double)*fft_size*mat_x_row));
    cudaMemset(d_padding_x, 0, sizeof(double)*fft_size*mat_x_row);
    CHECK(cudaMalloc((void**)&d_padding_y, sizeof(double)*fft_size*mat_y_row));
    cudaMemset(d_padding_y, 0, sizeof(double)*fft_size*mat_y_row);

    dim3 copyThreads(32, 32);
    dim3 copyBlocks_X((mat_x_row+copyThreads.x-1)/copyThreads.x, (mat_col+copyThreads.y-1)/copyThreads.y);
    dim3 copyBlocks_Y((mat_y_row+copyThreads.x-1)/copyThreads.x, (mat_col+copyThreads.y-1)/copyThreads.y);
    copyPaddingData2D<<<copyBlocks_X, copyThreads>>>(d_mat_x, d_padding_x, mat_x_row,
                                                     mat_col, fft_size);
    copyPaddingData2D<<<copyBlocks_Y, copyThreads>>>(d_mat_y, d_padding_y, mat_y_row,
                                                     mat_col, fft_size);
    cudaDeviceSynchronize();

    CHECK(cudaMalloc((void**)&o_xdata, sizeof(cufftDoubleComplex)*fft_size*mat_x_row));
    CHECK(cudaMalloc((void**)&o_ydata, sizeof(cufftDoubleComplex)*fft_size*mat_y_row));
    CHECK(cudaMalloc((void**)&o_data, sizeof(cufftDoubleComplex)*fft_size*mat_x_row*mat_y_row));

    cufftHandle fftPlanFwd_x;
    cufftHandle fftPlanFwd_y;
    cufftHandle fftPlanInv;
    cufftResult stat;

    stat = cufftPlanMany(&fftPlanFwd_x, RANK, n_x, inembed_x, 
                        istride, idist, onembed_x, 
                        ostride, odist, CUFFT_D2Z, mat_x_row);
    if (stat != CUFFT_SUCCESS) {printf("cufft MakePlan error %d\n",stat); exit( -1 );}            
    stat = cufftPlanMany(&fftPlanFwd_y, RANK, n_y, inembed_y, 
                         istride, idist, onembed_y, 
                         ostride, odist, CUFFT_D2Z, mat_y_row);
    if (stat != CUFFT_SUCCESS) {printf("cufft MakePlan error %d\n",stat); exit( -1 );}
    stat = cufftPlanMany(&fftPlanInv, RANK, n_out, inembed_out, 
                        istride, idist_out, onembed_out, 
                        ostride, odist_out, CUFFT_Z2D, mat_x_row*mat_y_row);
    if (stat != CUFFT_SUCCESS) {printf("cufft MakePlan error %d\n",stat); exit( -1 );}
    stat = cufftExecD2Z(fftPlanFwd_x, d_padding_x, o_xdata);
    if (stat != CUFFT_SUCCESS) {printf("cufft cufftExecD2Z error %d\n",stat); exit( -1 );}
    stat = cufftExecD2Z(fftPlanFwd_y, d_padding_y, o_ydata);
    if (stat != CUFFT_SUCCESS) {printf("cufft cufftExecD2Z error %d\n",stat); exit( -1 );}

    dim3 complexThreads(32, 2, 16);
    dim3 complexBlocks((mat_x_row+complexThreads.x-1)/complexThreads.x, 
                       (mat_y_row+complexThreads.y-1)/complexThreads.y,
                       (fft_size+complexThreads.z-1)/complexThreads.z);
    complex_dot3D<cufftDoubleComplex, double><<<complexBlocks, complexThreads>>>(o_xdata, o_ydata, o_data, sum_dot,
                                                     mat_x_row, mat_y_row, fft_size);
    cudaDeviceSynchronize();

    stat = cufftExecZ2D(fftPlanInv, o_data, fft_out);
    if (stat != CUFFT_SUCCESS) {printf("cufft cufftExecZ2D error %d\n",stat); exit( -1 );}

    cufftDestroy(fftPlanFwd_x);
    cufftDestroy(fftPlanFwd_y);
    cufftDestroy(fftPlanInv);
    cudaFree(d_padding_x);
    cudaFree(d_padding_y);
    cudaFree(o_xdata);
    cudaFree(o_ydata);
    cudaFree(o_data);
}


//return x11 x12 x13...x1k   x21 x22 x23...x2k ...... xn1 xn2...xnk
template<typename T>
void NCC_3D(const T *d_mat_x, const T *d_mat_y, T *d_mat_out, 
            const int mat_x_row, const int mat_y_row,
            const int mat_col)
{
    const int fft_size = 1 << (int)ceil(my_log2((2*mat_col - 1)));
    T *d_copy_x;
    T *d_copy_y;
    T *d_mat_tmp;
    T *sum_dot;
    T *d_fft_out;
    const int blockSize = 256;
    const int max_row = mat_x_row>mat_y_row?mat_x_row:mat_y_row;
    const int gridSize = (max_row + blockSize - 1) / blockSize;
    bool flag = mat_x_row>mat_y_row?true:false;
    
    CHECK(cudaMalloc((void**)&d_copy_x, sizeof(T)*mat_x_row*mat_col));
    CHECK(cudaMalloc((void**)&d_copy_y, sizeof(T)*mat_y_row*mat_col));
    CHECK(cudaMalloc((void**)&d_mat_tmp, sizeof(T)*(mat_y_row+mat_x_row)));
    CHECK(cudaMalloc((void**)&sum_dot, sizeof(T)*(mat_y_row*mat_x_row)));
    CHECK(cudaMalloc((void**)&d_fft_out, sizeof(T)*(mat_x_row*mat_y_row*fft_size)));
    
    CHECK(cudaMemcpy(d_copy_x, d_mat_x, sizeof(T)*mat_x_row*mat_col, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(d_copy_y, d_mat_y, sizeof(T)*mat_y_row*mat_col, cudaMemcpyDeviceToDevice));

    //get norm dot
    Mat_norm<<<gridSize, blockSize>>>(d_copy_x, d_copy_y, d_mat_tmp, mat_x_row, mat_y_row, mat_col);
    cudaDeviceSynchronize();
    
    Mat_norm_dot<<<gridSize, blockSize>>>(d_mat_tmp, sum_dot, mat_x_row, mat_y_row, flag);
    cudaDeviceSynchronize();
    
    //傅里叶变换
    excu_fft3D<T>(d_mat_x, d_mat_y, d_fft_out, sum_dot, mat_x_row, mat_y_row, mat_col);

    const int dest_col = 2*mat_col-1;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((mat_x_row*mat_y_row+threadsPerBlock.x-1)/threadsPerBlock.x, 
                   (dest_col+threadsPerBlock.y-1)/threadsPerBlock.y);
    copy_result2D<<<numBlocks, threadsPerBlock>>>(d_fft_out, d_mat_out, fft_size, dest_col, mat_x_row*mat_y_row);
    cudaDeviceSynchronize();

    cudaFree(d_copy_x);
    cudaFree(d_copy_y);
    cudaFree(d_mat_tmp);
    cudaFree(sum_dot);
    cudaFree(d_fft_out);

}


#endif
