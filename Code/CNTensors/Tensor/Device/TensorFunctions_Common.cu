//=============================================================================
// FILENAME : TensorFunctions_Common.cu
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [29/05/2020 nbale]
//=============================================================================
#include "CNTensorsPch.h"

#define __KERNALCAL(funcName, ...) \
    if (byIndexCount > _kSmallOrder) \
    { \
        funcName << <uiBlock, uiThread >> > (__VA_ARGS__); \
    } \
    else \
    { \
        funcName##_Small << <uiBlock, uiThread >> > (__VA_ARGS__); \
    }


#define __BuildMultiplyLength(ptr) \
    UINT* mul_length = appGetTensorOpWorkingSpace()->GetMultiplyLengthBuffer(); \
    mul_length[byIndexCount - 1] = 1; \
    for (INT i = byIndexCount - 2; i >= 0; --i) /* do not use BYTE here*/ \
    { \
        mul_length[i] = mul_length[i + 1] * lengths[i + 1]; \
    } \
    _memcpy_hd(ptr, mul_length, dataSize);


__BEGIN_NAMESPACE

#pragma region kernels

template <class T>
__global__ void _CN_LAUNCH_BOUND _kernel_BlockCopyMasked(
    T* dst,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ dstIndexStart,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxDst = IndexWithStrideWithStart(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    const UINT uiIdxSrc = IndexWithStrideWithStart(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    const UINT uiIdxMask = (NULL == mask) ? 0 : IndexWithStrideWithStart(maskStride, maskIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    if (NULL == mask || 0 != mask[uiIdxMask])
    {
        dst[uiIdxDst] = src[uiIdxSrc];
    }
}

template <class T>
__global__ void _CN_LAUNCH_BOUND _kernel_BlockCopyMasked_Small(
    T* dst,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ dstIndexStart,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const SWorkingIndex idx = ThreadIndexToWorkIndex(threadIdx.x + blockIdx.x * blockDim.x, mutipliedlengths, byIndexCount);
    const UINT uiIdxDst = WorkIndexToTensorIndexWithStart(idx.m_Idx, dstStride, dstIndexStart, byIndexCount);
    const UINT uiIdxSrc = WorkIndexToTensorIndexWithStart(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);
    const UINT uiIdxMask = (NULL == mask) ? 0 : WorkIndexToTensorIndexWithStart(idx.m_Idx, maskStride, maskIndexStart, byIndexCount);
    if (NULL == mask || 0 != mask[uiIdxMask])
    {
        dst[uiIdxDst] = src[uiIdxSrc];
    }
}

template <class T>
__global__ void _CN_LAUNCH_BOUND _kernel_FillMasked(
    T* src, T val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxSrc = IndexWithStrideWithStart(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    const UINT uiIdxMask = (NULL == mask) ? 0 : IndexWithStrideWithStart(maskStride, maskIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    if (NULL == mask || 0 != mask[uiIdxMask])
    {
        src[uiIdxSrc] = val;
    }
}

template <class T>
__global__ void _CN_LAUNCH_BOUND _kernel_FillMasked_Small(
    T* src, T val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const SWorkingIndex idx = ThreadIndexToWorkIndex(threadIdx.x + blockIdx.x * blockDim.x, mutipliedlengths, byIndexCount);
    const UINT uiIdxSrc = WorkIndexToTensorIndexWithStart(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);
    const UINT uiIdxMask = (NULL == mask) ? 0 : WorkIndexToTensorIndexWithStart(idx.m_Idx, maskStride, maskIndexStart, byIndexCount);
    if (NULL == mask || 0 != mask[uiIdxMask])
    {
        src[uiIdxSrc] = val;
    }
}

template <class T>
__global__ void _CN_LAUNCH_BOUND _kernel_Transpose(
    T* dst,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcMultiplyLength,
    BYTE byIndexCountX)
{
    const UINT uiThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxSrc = IndexWithStride(srcStride, srcMultiplyLength, threadIdx.x + blockIdx.x * blockDim.x, byIndexCountX);
    dst[uiThreadIdx] = src[uiIdxSrc];
}

template <class T>
__global__ void _CN_LAUNCH_BOUND _kernel_Transpose_Small(
    T* dst,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcMultiplyLength,
    BYTE byIndexCountX)
{
    const UINT uiThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const SWorkingIndex idx = ThreadIndexToWorkIndex(uiThreadIdx, srcMultiplyLength, byIndexCountX);
    const UINT uiIdxSrc = WorkIndexToTensorIndex(idx.m_Idx, srcStride, byIndexCountX);
    dst[uiThreadIdx] = src[uiIdxSrc];
}

#pragma endregion

template <class T> __DLL_EXPORT void BlockCopyMasked(
    T* dst,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ dstIndexStart,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 7;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetTensorOpWorkingSpace()->GetSmallDeviceBuffer(totalBufferSize);
    _memcpy_hd(deviceBuffer, dstStride, dataSize);
    _memcpy_hd(deviceBuffer + dataSize, srcStride, dataSize);
    if (NULL != maskStride)
    {
        _memcpy_hd(deviceBuffer + dataSize * 5, maskStride, dataSize);
    }
    if (NULL != dstIndexStart)
    {
        _memcpy_hd(deviceBuffer + dataSize * 2, dstIndexStart, dataSize);
    }
    if (NULL != srcIndexStart)
    {
        _memcpy_hd(deviceBuffer + dataSize * 3, srcIndexStart, dataSize);
    }
    if (NULL != maskIndexStart)
    {
        _memcpy_hd(deviceBuffer + dataSize * 6, maskIndexStart, dataSize);
    }

    __BuildMultiplyLength(deviceBuffer + dataSize * 4);

    __KERNALCAL(_kernel_BlockCopyMasked,
        dst,
        (UINT*)deviceBuffer,
        (NULL != dstIndexStart) ? (UINT*)(deviceBuffer + dataSize * 2) : appGetTensorOpWorkingSpace()->GetZeroStartBuffer(),
        src,
        (UINT*)(deviceBuffer + dataSize),
        (NULL != srcIndexStart) ? (UINT*)(deviceBuffer + dataSize * 3) : appGetTensorOpWorkingSpace()->GetZeroStartBuffer(),
        mask,
        (UINT*)(deviceBuffer + dataSize * 5),
        (NULL != maskIndexStart) ? (UINT*)(deviceBuffer + dataSize * 6) : appGetTensorOpWorkingSpace()->GetZeroStartBuffer(),
        (UINT*)(deviceBuffer + dataSize * 4),
        byIndexCount
        );
}

template <class T> __DLL_EXPORT void FillMasked(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 5;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetTensorOpWorkingSpace()->GetSmallDeviceBuffer(totalBufferSize);
    _memcpy_hd(deviceBuffer, srcStride, dataSize);
    if (NULL != maskStride)
    {
        _memcpy_hd(deviceBuffer + dataSize * 3, maskStride, dataSize);
    }
    if (NULL != srcIndexStart)
    {
        _memcpy_hd(deviceBuffer + dataSize, srcIndexStart, dataSize);
    }
    if (NULL != maskIndexStart)
    {
        _memcpy_hd(deviceBuffer + dataSize * 4, maskIndexStart, dataSize);
    }

    __BuildMultiplyLength(deviceBuffer + dataSize * 2);

    __KERNALCAL(_kernel_FillMasked,
        src,
        val,
        (UINT*)(deviceBuffer),
        (NULL != srcIndexStart) ? (UINT*)(deviceBuffer + dataSize) : appGetTensorOpWorkingSpace()->GetZeroStartBuffer(),
        mask,
        (UINT*)(deviceBuffer + dataSize * 3),
        (NULL != maskIndexStart) ? (UINT*)(deviceBuffer + dataSize * 4) : appGetTensorOpWorkingSpace()->GetZeroStartBuffer(),
        (UINT*)(deviceBuffer + dataSize * 2),
        byIndexCount);
}

template <class T> __DLL_EXPORT void Random(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    
}


template <class T> __DLL_EXPORT void RandomMasked(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcStart,
    const UINT* __restrict__ srcEnd,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    
}

template <class T> __DLL_EXPORT void Transpose(T* dst,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount)
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 5;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetTensorOpWorkingSpace()->GetSmallDeviceBuffer(totalBufferSize);
    _memcpy_hd(deviceBuffer, srcStride, dataSize);

    __BuildMultiplyLength(deviceBuffer + dataSize);

    __KERNALCAL(_kernel_Transpose,
        dst,
        src,
        (UINT*)deviceBuffer,
        (UINT*)(deviceBuffer + dataSize),
        byIndexCount);
}

template <class T> __DLL_EXPORT void DebugPrint(
    const T* __restrict__ src,
    UINT uiSize)
{
    T* hostBuffer = (T*)malloc(sizeof(T) * uiSize);
    _memcpy_dh(hostBuffer, src, sizeof(T) * uiSize);

    for (UINT i = 0; i < uiSize; ++i)
    {
        appGeneral(_T("%d: "), i);
        LogValue(hostBuffer[i]);
        appGeneral(_T("\n"));
    }

    appSafeFree(hostBuffer);
}

template <class T> __DLL_EXPORT void DebugPrint(
    const T* __restrict__ src,
    UINT uiXDim,
    UINT uiYDim)
{
    const UINT uiSize = uiXDim * uiYDim;
    T* hostBuffer = (T*)malloc(sizeof(T) * uiSize);
    _memcpy_dh(hostBuffer, src, sizeof(T) * uiSize);

    for (UINT x = 0; x < uiXDim; ++x)
    {
        for (UINT y = 0; y < uiYDim; ++y)
        {
            LogValue(hostBuffer[x * uiYDim + y]);
            if (y != uiYDim - 1)
            {
                appGeneral(_T(", "));
            }
        }
        appGeneral(_T("\n"));
    }

    appSafeFree(hostBuffer);
}


__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
