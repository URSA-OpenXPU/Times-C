// Define this to turn on error checking
#ifndef CUDA_ERROR_CHECK
#define CUDA_ERROR_CHECK

#define CHECK( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{

    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }


    return;
}

#endif