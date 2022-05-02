//=============================================================================
// FILENAME : CudaIncludes.h
// 
// DESCRIPTION:
// This is the file for some common CUDA usage
//
// REVISION:
//  [31/05/2020 nbale]
//=============================================================================

#ifndef _CUDAINCLUDES_H_
#define _CUDAINCLUDES_H_

#include "cuda_runtime.h"
#include "vector_types.h"

#include "cuda.h"
#include "math.h"
#include "cuComplex.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cufft.h>

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET appCudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif


#define _FAIL_EXIT \
appFlushLog(); \
DEVICE_RESET \
exit(EXIT_FAILURE);

__BEGIN_NAMESPACE

extern CNAPI void appCudaDeviceReset();
extern CNAPI const ANSICHAR* appCudaGetErrorName(cudaError_t error);

__END_NAMESPACE

#endif //#ifndef _CUDAINCLUDES_H_

//=============================================================================
// END OF FILE
//=============================================================================