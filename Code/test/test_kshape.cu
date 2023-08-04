#include <iostream>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "kshape.cuh"
#include "readfile.cuh"
#include <algorithm>
#include <string>
using namespace std;
typedef float types;
#define N 5
#define ROWS 6

void labeled_process(string file_path,string name, int iter_chose)
{
    GpuTimer timer;

    vector<int> label;
    int mat_rows,mat_cols;
    types *mat = readFile<types>(file_path, label, mat_rows, mat_cols);
    cout<<"mat_rows"<<mat_rows<<" "<<"mat_cols"<<mat_cols<<endl;
    int iternum = 10;
    int iternum2 = iternum;
    
    auto maxPosition = max_element(label.begin(), label.end());
    auto minPosition = min_element(label.begin(), label.end());
    int k = *maxPosition;
    if(*minPosition == 0)
        k += 1;
    else if(*minPosition == -1)
        k = 2;

    types *d_mat;
    int *idx = new int[mat_rows];
    timer.Start();
    cudaMalloc((void**)&d_mat, sizeof(types)*mat_rows*mat_cols);
    cudaMemcpy(d_mat, mat, sizeof(types)*mat_rows*mat_cols, cudaMemcpyHostToDevice);
    z_norm_gpu_x(d_mat, mat_rows, mat_cols, 1);
    timer.Stop();
    printf("first norm run on GPU: %f msecs.\n", timer.Elapsed());
    
    timer.Start();
    int sum_class;
    int cal_select = 0;
    if(iter_chose == 1)
    {
        cout<<"exec k_shape_tsl"<<endl;
        k_shape_tsl(d_mat, mat, idx, mat_rows, mat_cols, k, iternum, sum_class, cal_select);
    }   
    else
    {
        cout<<"exec k_shape"<<endl;
        k_shape(d_mat, idx, mat_rows, mat_cols, k, iternum, sum_class, cal_select);
    }  
    if(sum_class == 1 || sum_class<(k/2))
    {
        cal_select = 1;
        sum_class = k;
        k_shape(d_mat, idx, mat_rows, mat_cols, k, iternum2, sum_class, cal_select);
    }
    timer.Stop();
    float time_cost =  timer.Elapsed();
    printf("total time run on GPU: %f msecs.\n", time_cost);
    float time_avg = time_cost/iternum/1000;

    string run_file[3];
    run_file[0] = file_path;
    run_file[1] = to_string(time_avg);
    run_file[2] = to_string(k);

    //string out_path = "../out/cukshape/"+name+".txt";
    string out_path = "/home/songruibao/test_git/Times-C/Code/out/cukshape/result.txt";
    writeFile(out_path, idx, mat_rows);

    cudaFree(d_mat);
    delete[] idx;
    delete[] mat;
    cout<<endl;
}


void unlabeled_process(string file_path, const int k, string name, int iter_chose)
{
    
    GpuTimer timer;
    int mat_rows,mat_cols;
    
    types *mat = read_unlabeled_File<types>(file_path, mat_rows, mat_cols);
    
    cout<<"mat_rows"<<mat_rows<<" "<<"mat_cols"<<mat_cols<<endl;
    int iternum = 20;
    int iternum2 = iternum;

    types *d_mat;
    int *idx = new int[mat_rows];
    
    timer.Start();
    cudaMalloc((void**)&d_mat, sizeof(types)*mat_rows*mat_cols);
    cudaMemcpy(d_mat, mat, sizeof(types)*mat_rows*mat_cols, cudaMemcpyHostToDevice);
    z_norm_gpu_x(d_mat, mat_rows, mat_cols, 1);
    int sum_class = k;
    int cal_select = 0;
    if(iter_chose == 1)
    {
        cout<<"exec k_shape_tsl"<<endl;
        k_shape_tsl(d_mat, mat, idx, mat_rows, mat_cols, k, iternum, sum_class, cal_select);
    }   
    else
    {
        cout<<"exec k_shape"<<endl;
        k_shape(d_mat, idx, mat_rows, mat_cols, k, iternum, sum_class, cal_select);
    }    
    if(sum_class == 1 || sum_class<(k/2))
    {
        cal_select = 1;
        sum_class = k;
        k_shape(d_mat, idx, mat_rows, mat_cols, k, iternum2, sum_class, cal_select);
    }
        
    timer.Stop();
    float time_cost =  timer.Elapsed();
    printf("total time run on GPU: %f msecs.\n", time_cost);
    float time_avg = time_cost/iternum/1000;

    string run_file[3];
    run_file[0] = file_path;
    run_file[1] = to_string(time_avg);
    run_file[2] = to_string(k);

    //string out_path = "../out/cukshape/"+name+".txt";
    string out_path = "/home/songruibao/test_git/Times-C/Code/out/cukshape/result.txt";
    writeFile(out_path, idx, mat_rows);

    cudaFree(d_mat);
    delete[] idx;
    delete[] mat;
    cout<<endl;
}


void testfunc(string fname, int k, int flag, int iter_chose)
{
    cout<<fname<<endl;
    if(flag)
        unlabeled_process(fname, k, "result", iter_chose);
    else
        labeled_process(fname, "result", iter_chose);
}


// ./test -d 0(device name) filePath class_number label_or_unlabel_flag 
//./test -d 0 /home/songruibao/code/data/InsectSound/InsectSound_TRAIN 10 1
int main(int argc, char ** argv)
{
    int device_id;
    char const *filename;
    device_id = 0;
    int argi = 1;
    int k;
    char *devstr;
    int flag;
    int iter_chose;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);

        // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    size_t size = min( int(deviceProp.l2CacheSize * 0.80) , deviceProp.persistingL2CacheMaxSize );
    cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size); 

    printf("---------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n",
            device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);
    if(argc > argi)
    {        
        filename = argv[argi];
        printf("-------------- %s --------------\n", filename);
        argi++;
    }
    
    if(argc > argi)
    {        
        k = atoi(argv[argi]);
        argi++;
    }

    if(argc > argi)
    {        
        flag = atoi(argv[argi]);
        argi++;
    }

    if(argc > argi)
    {        
        iter_chose = atoi(argv[argi]);
        argi++;
    }

    string fname = filename;
    testfunc(fname, k, flag, iter_chose);
}