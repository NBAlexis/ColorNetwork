//=============================================================================
// FILENAME : CudaHelper.h
// 
// DESCRIPTION:
// This is the file for CUDA Testing usage
//
// REVISION:
//  [12/3/2018 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

void appCudaDeviceReset()
{
    cudaDeviceReset();
}

const ANSICHAR* appCudaGetErrorName(cudaError_t error)
{
    return cudaGetErrorName(error);
}

void _appCudaFree(void* ptr)
{
    if (NULL == appGetCuda())
    {
        checkCudaErrors(cudaFree(ptr));
    }
    else
    {
        appGetCuda()->Free(ptr);
    }
}

__constant__ INT _constIntegers[kContentLength];
//__constant__ Real _constFloats[kContentLength];
__constant__ CRandom* __r;


#pragma region Kernels

//template<class Operator>
//__global__ void _CN_LAUNCH_BOUND_SINGLE
//_kernelDebugFunction(TOperator_DS<Operator, _SComplex, FLOAT>* pDS)
//{
//    FLOAT b = 3.0;
//    _SComplex res = pDS->Do(b);
//    printf("===============\n");
//    printf("%f, %f", res.x, res.y);
//}

template<class T> __global__ void _CN_LAUNCH_BOUND_MAXTHREAD
_kernelThreadBufferZero(T * arr, T initial)
{
    arr[threadIdx.x + blockIdx.x * blockDim.x] = initial;
}

template<class T> __global__ void _CN_LAUNCH_BOUND_MAXTHREAD
_kernelReduceSum(T* arr, UINT uiJump, UINT uiMax)
{
    //for length 16 array
    //for jump = 1, this is 1->0, 3->2, 5->4, 7->6, 9->10, 11->10, 13->12, 15->14 
    //for jump = 2, this is 2->0, 6->4, 10->8, 14->12 
    //for jump = 4, this is 4->0, 12->8 
    //for jump = 8, this is 8->0, and is finished.

    //id target = idx * (jump << 1)
    //id from = target + jump
    UINT uiIdFrom = (threadIdx.x + blockIdx.x * blockDim.x) * (uiJump << 1) + uiJump;
    if (uiIdFrom < uiMax)
    {
        arr[uiIdFrom - uiJump] = _Add(arr[uiIdFrom - uiJump], arr[uiIdFrom]);
    }
}

template<class T> __global__ void _CN_LAUNCH_BOUND_MAXTHREAD
_kernelReduceProd(T* arr, UINT uiJump, UINT uiMax)
{
    //for length 16 array
    //for jump = 1, this is 1->0, 3->2, 5->4, 7->6, 9->10, 11->10, 13->12, 15->14 
    //for jump = 2, this is 2->0, 6->4, 10->8, 14->12 
    //for jump = 4, this is 4->0, 12->8 
    //for jump = 8, this is 8->0, and is finished.

    //id target = idx * (jump << 1)
    //id from = target + jump
    UINT uiIdFrom = (threadIdx.x + blockIdx.x * blockDim.x) * (uiJump << 1) + uiJump;
    if (uiIdFrom < uiMax)
    {
        arr[uiIdFrom - uiJump] = _Mul(arr[uiIdFrom - uiJump], arr[uiIdFrom]);
    }
}

#pragma endregion

CCudaHelper::~CCudaHelper()
{
    //Clean up constants

    ReleaseTemeraryBuffers();
    ReleaseHelpers();

    TArray<PTRINT> keys = m_pMemRecord.GetAllKeys();
    for (INT i = 0; i < keys.Num(); ++i)
    {
        appGeneral(_T("Used: size %llu, location %s\n"), m_pMemRecord[keys[i]].m_uiSize, m_pMemRecord[keys[i]].m_sInfo);
    }
}

#pragma region System Info

void CCudaHelper::DeviceQuery()
{
    appGeneral(_T(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n"));

    INT deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) 
    {
        appGeneral(_T("cudaGetDeviceCount returned %d\n-> %s\n"),
            static_cast<INT>(error_id), 
            cudaGetErrorString(error_id)); //TODO Unicode support
        appCrucial(_T("Result = FAIL\n"));
        _FAIL_EXIT;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) 
    {
        appGeneral(_T("There are no available device(s) that support CUDA\n"));
    }
    else 
    {
        appGeneral(_T("Detected %d CUDA Capable device(s)\n"), deviceCount);
    }

    INT dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) 
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        //TODO Unicode support
        appGeneral(_T("\nDevice %d: \"%s\"\n"), dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        appGeneral(_T("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n"),
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        appGeneral(_T("  CUDA Capability Major/Minor version number:    %d.%d\n"),
            deviceProp.major, deviceProp.minor);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        appGeneral(_T("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n"),
            static_cast<FLOAT>(deviceProp.totalGlobalMem / 1048576.0f),
            (QWORD)deviceProp.totalGlobalMem);
#else
        appGeneral(_T("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n"),
            static_cast<FLOAT>(deviceProp.totalGlobalMem / 1048576.0f),
            (QWORD)deviceProp.totalGlobalMem);
#endif

        appGeneral(_T("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n"),
            deviceProp.multiProcessorCount,
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
            deviceProp.multiProcessorCount);
        appGeneral(
            _T("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n"),
            deviceProp.clockRate * 1e-3f, 
            deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000

        //No plan B for CUDA < 5.0 since we do not support it
        // This is supported in CUDA 5.0 (runtime API device properties)
        appGeneral(_T("  Memory Clock rate:                             %.0f Mhz\n"),
            deviceProp.memoryClockRate * 1e-3f);
        appGeneral(_T("  Memory Bus Width:                              %d-bit\n"),
            deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize) 
        {
            appGeneral(_T("  L2 Cache Size:                                 %d bytes\n"),
                deviceProp.l2CacheSize);
        }

#endif

        appGeneral(_T("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n"),
            deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
            deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
            deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        appGeneral(_T("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n"),
            deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        appGeneral(
            _T("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n"),
            deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);

        appGeneral(_T("  Total amount of constant memory:               %lu bytes\n"),
            deviceProp.totalConstMem);
        appGeneral(_T("  Total amount of shared memory per block:       %lu bytes\n"),
            deviceProp.sharedMemPerBlock);
        appGeneral(_T("  Total number of registers available per block: %d\n"),
            deviceProp.regsPerBlock);
        appGeneral(_T("  Warp size:                                     %d\n"),
            deviceProp.warpSize);
        appGeneral(_T("  Maximum number of threads per multiprocessor:  %d\n"),
            deviceProp.maxThreadsPerMultiProcessor);
        appGeneral(_T("  Maximum number of threads per block:           %d\n"),
            deviceProp.maxThreadsPerBlock);
        appGeneral(_T("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n"),
            deviceProp.maxThreadsDim[0], 
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        appGeneral(_T("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n"),
            deviceProp.maxGridSize[0], 
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
        appGeneral(_T("  Maximum memory pitch:                          %lu bytes\n"),
            deviceProp.memPitch);
        appGeneral(_T("  Texture alignment:                             %lu bytes\n"),
            deviceProp.textureAlignment);
        appGeneral(_T("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n"),
            (deviceProp.deviceOverlap ? _T("Yes") : _T("No")),
            deviceProp.asyncEngineCount);
        appGeneral(_T("  Run time limit on kernels:                     %s\n"),
            deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        appGeneral(_T("  Integrated GPU sharing Host Memory:            %s\n"),
            deviceProp.integrated ? "Yes" : "No");
        appGeneral(_T("  Support host page-locked memory mapping:       %s\n"),
            deviceProp.canMapHostMemory ? "Yes" : "No");
        appGeneral(_T("  Alignment requirement for Surfaces:            %s\n"),
            deviceProp.surfaceAlignment ? "Yes" : "No");
        appGeneral(_T("  Device has ECC support:                        %s\n"),
            deviceProp.ECCEnabled ? _T("Enabled") : _T("Disabled"));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        appGeneral(_T("  CUDA Device Driver Mode (TCC or WDDM):         %s\n"),
            deviceProp.tccDriver ? _T("TCC (Tesla Compute Cluster Driver)")
            : _T("WDDM (Windows Display Driver Model)"));
#endif
        appGeneral(_T("  Device supports Unified Addressing (UVA):      %s\n"),
            deviceProp.unifiedAddressing ? _T("Yes") : _T("No"));
        appGeneral(_T("  Device supports Compute Preemption:            %s\n"),
            deviceProp.computePreemptionSupported ? _T("Yes") : _T("No"));
        appGeneral(_T("  Supports Cooperative Kernel Launch:            %s\n"),
            deviceProp.cooperativeLaunch ? _T("Yes") : _T("No"));
        appGeneral(_T("  Supports MultiDevice Co-op Kernel Launch:      %s\n"),
            deviceProp.cooperativeMultiDeviceLaunch ? _T("Yes") : _T("No"));
        appGeneral(_T("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n"),
            deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const TCHAR *sComputeMode[] = {
            _T("Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)"),
            _T("Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)"),
            _T("Prohibited (no host thread can use ::cudaSetDevice() with this device)"),
            _T("Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)"),
            _T("Unknown"),
            NULL };
        appGeneral(_T("  Compute Mode:\n"));
        appGeneral(_T("     < %s >\n"), sComputeMode[deviceProp.computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2) 
    {
        cudaDeviceProp prop[64];
        INT gpuid[64];  // we want to find the first two GPUs that can support P2P
        INT gpu_p2p_count = 0;

        for (INT i = 0; i < deviceCount; ++i) 
        {
            checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)) 
            {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows
                // must be enabled to support this
                if (!prop[i].tccDriver)
                {
                    appGeneral(_T("GPU:%d is presented but NOT Tesla Compute Cluster driver!\n"), i);
                    continue;
                }
#endif
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
            else
            {
                appGeneral(_T("GPU:%d is presented but NOT support P2P!\n"), i);
            }
        }

        // Show all the combinations of support P2P GPUs
        INT can_access_peer;

        if (gpu_p2p_count >= 2) 
        {
            for (INT i = 0; i < gpu_p2p_count; i++) 
            {
                for (INT j = 0; j < gpu_p2p_count; j++) 
                {
                    if (gpuid[i] == gpuid[j]) 
                    {
                        continue;
                    }
                    checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                    appGeneral(_T("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n"),
                        prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                        can_access_peer ? _T("Yes") : _T("No"));
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    appGeneral(_T("\n"));
    CCString sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    appSprintf(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
    appSprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
        (driverVersion % 100) / 10);
#endif
    sProfileString += cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    appSprintf(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
    appSprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
        (runtimeVersion % 100) / 10);
#endif
    sProfileString += cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    appSprintf(cTemp, 10, "%d", deviceCount);
#else
    appSprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
    sProfileString += cTemp;
    sProfileString += "\n";
    appGeneral(_T("%s"), sProfileString.c_str()); //TODO Unicode support

    appGeneral(_T("Result = PASS\n"));

}

void CCudaHelper::MemoryQuery()
{
    size_t availableMemory, totalMemory;
    cudaMemGetInfo(&availableMemory, &totalMemory);
    const size_t usedMemory = totalMemory - availableMemory;
    appGeneral(_T("Device Memory: used %llu, available %llu, total %llu\n"), usedMemory, availableMemory, totalMemory);

    TArray<PTRINT> keys = m_pMemRecord.GetAllKeys();
    for (INT i = 0; i < keys.Num(); ++i)
    {
        appGeneral(_T("Used: size %llu, location %s\n"), m_pMemRecord[keys[i]].m_uiSize, m_pMemRecord[keys[i]].m_sInfo);
    }
}

void CCudaHelper::DebugFunction()
{
    //TOperatorDS_Sin<_SComplex, FLOAT> op;
    //TOperatorDS_Sin<_SComplex, FLOAT>* device = NULL;
    //checkCudaErrors(cudaMalloc((void**)&device, sizeof(TOperatorDS_Sin<_SComplex, FLOAT>)));
    //checkCudaErrors(cudaMemcpy(device, &op, sizeof(TOperatorDS_Sin<_SComplex, FLOAT>), cudaMemcpyHostToDevice));
    //_kernelDebugFunction << <1,1 >> > (device);
    //checkCudaErrors(cudaDeviceSynchronize());
}

#pragma endregion

#pragma region fast sum and prod

TArray<UINT> CCudaHelper::GetMaxThreadCountAndThreadPerblock()
{
    TArray<UINT> ret;

    INT deviceCount = 0;
    const cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        appCrucial("cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<INT>(error_id), cudaGetErrorString(error_id));
        appCrucial("Result = FAIL\n");
        _FAIL_EXIT;
    }

    if (0 == deviceCount)
    {
        appCrucial(_T("This program need GPU but you do NOT have a GPU.\n"));
        _FAIL_EXIT;
    }

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    ret.AddItem(deviceProp.maxThreadsPerBlock);
    ret.AddItem(deviceProp.maxThreadsDim[0]);
    ret.AddItem(deviceProp.maxThreadsDim[1]);
    ret.AddItem(deviceProp.maxThreadsDim[2]);

    return ret;
}

void CCudaHelper::AllocateTemeraryBuffers()
{
    //checkCudaErrors(cudaMalloc((void**)&m_pSmallSizeBuffer, kSmallBufferSize));
    //checkCudaErrors(cudaMalloc((void**)&m_pRealBufferThreadCount, sizeof(Real)* MAX_THREAD));
    //checkCudaErrors(cudaMalloc((void**)&m_pComplexBufferThreadCount, sizeof(CNComplex)* MAX_THREAD));
    //checkCudaErrors(cudaMalloc((void**)&m_pIntBufferThreadCount, sizeof(INT) * MAX_THREAD));
}

void CCudaHelper::ReleaseTemeraryBuffers()
{
    //if (NULL != m_pSmallSizeBuffer)
    //{
    //    checkCudaErrors(cudaFree(m_pSmallSizeBuffer));
    //    m_pSmallSizeBuffer = NULL;
    //}

    //_freed(m_pComplexBufferThreadCount);
    //_freed(m_pIntBufferThreadCount);
}

void CCudaHelper::_Malloc(const TCHAR* sLocation, void** ptr, UINT uiSize, UINT uiReason)
{
    checkCudaErrors(cudaMalloc(ptr, uiSize));

    CudaMemInfo newInfo;
    newInfo.m_uiSize = uiSize;
    newInfo.m_uiReason = uiReason;
    newInfo.m_pPointer = (PTRINT)(*ptr);
    appStrcpy(newInfo.m_sInfo, 256, sLocation);
    assert(!m_pMemRecord.Exist(newInfo.m_pPointer));
    m_pMemRecord.SetAt(newInfo.m_pPointer, newInfo);
}

void CCudaHelper::Free(void* ptr)
{
    if (NULL != ptr)
    {
        PTRINT byPtr = (PTRINT)ptr;
        checkCudaErrors(cudaFree(ptr));
        assert(m_pMemRecord.Exist(byPtr));
        m_pMemRecord.RemoveKey(byPtr);
    }
}

void CCudaHelper::CopyDD(void* dest, const void* src, UINT uiSize)
{
    checkCudaErrors(cudaMemcpy(dest, src, uiSize, cudaMemcpyDeviceToDevice));
}

void CCudaHelper::CopyHD(void* dest, const void* src, UINT uiSize)
{
    checkCudaErrors(cudaMemcpy(dest, src, uiSize, cudaMemcpyHostToDevice));
}

void CCudaHelper::CopyDH(void* dest, const void* src, UINT uiSize)
{
    checkCudaErrors(cudaMemcpy(dest, src, uiSize, cudaMemcpyDeviceToHost));
}

template<class T> T CCudaHelper::ReduceSum(T* deviceBuffer, UINT uiLength)
{
    const UINT iPower = GetReduceDim(uiLength) + 1;
    for (UINT i = 0; i <= iPower; ++i)
    {
        const UINT iJump = 1 << i;
        const UINT iThreadNeeded = 1 << (iPower - i);
        const UINT iBlock = iThreadNeeded > MAX_THREAD ? iThreadNeeded / MAX_THREAD : 1;
        const UINT iThread = iThreadNeeded > MAX_THREAD ? MAX_THREAD : iThreadNeeded;
        _kernelReduceSum << <iBlock, iThread >> > (deviceBuffer, iJump, uiLength);
    }
    T result[1];
    _memcpy_dh(result, deviceBuffer, sizeof(T));
    return result[0];
}

template<class T> T CCudaHelper::ReduceProd(T* deviceBuffer, UINT uiLength)
{
    const UINT iPower = GetReduceDim(uiLength) + 1;
    for (UINT i = 0; i <= iPower; ++i)
    {
        const UINT iJump = 1 << i;
        const UINT iThreadNeeded = 1 << (iPower - i);
        const UINT iBlock = iThreadNeeded > MAX_THREAD ? iThreadNeeded / MAX_THREAD : 1;
        const UINT iThread = iThreadNeeded > MAX_THREAD ? MAX_THREAD : iThreadNeeded;
        _kernelReduceProd << <iBlock, iThread >> > (deviceBuffer, iJump, uiLength);
    }
    T result[1];
    _memcpy_dh(result, deviceBuffer, sizeof(T));
    return result[0];
}

template<class T> void CCudaHelper::ThreadBufferInitial(T* deviceBuffer, const T& val, UINT uiLength)
{
    _kernelThreadBufferZero << <1, uiLength >> > (deviceBuffer, val);
}

#pragma endregion

#pragma region Constants

void CCudaHelper::CopyConstants() const
{
    //checkCudaErrors(cudaMemcpyToSymbol(_constIntegers, m_ConstIntegers, sizeof(INT) * kContentLength));
    //checkCudaErrors(cudaMemcpyToSymbol(_constFloats, m_ConstFloats, sizeof(Real) * kContentLength));
}

void CCudaHelper::CopyRandomPointer(const class CRandom* r) const
{
    checkCudaErrors(cudaMemcpyToSymbol(__r, &r, sizeof(CRandom*)));
}

#pragma endregion

#pragma region Other helpers

void CCudaHelper::InitialHelpers()
{
    //m_pTensorWorkingSpace = new CTensorOpWorkingSpace();
}

void CCudaHelper::ReleaseHelpers()
{
    //appSafeDelete(m_pTensorWorkingSpace);
}

#pragma endregion

__END_NAMESPACE


//=============================================================================
// END OF FILE
//=============================================================================