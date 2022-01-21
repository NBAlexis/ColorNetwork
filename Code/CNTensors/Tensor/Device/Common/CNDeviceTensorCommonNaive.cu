//=============================================================================
// FILENAME : CNDeviceTensorCommonNaive.cu
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [19/06/2021 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

//__CN_FORCEOBJ_CPP(CNDeviceTensorCommonNaive);

//This is the critical specialization
template class CNDeviceTensorCommonNaive<_SComplex>;

#pragma region kernels

template <class Operator, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveOneOperator(
    TOperator_D<Operator, srcT> op,
    srcT* src,
    const UINT* __restrict__ srcStride,
    UINT srcIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    op.Do(src[uiIdxSrc]);
}

template <class Operator, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveOneOperator_Small(
    TOperator_D<Operator, srcT> op,
    srcT* src,
    const UINT* __restrict__ srcStride,
    UINT srcIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
    const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);
    op.Do(src[uiIdxSrc]);
}

template <class Operator, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveTwoOperator_DS_Value(
    TOperator_DS<Operator, dstT, srcT> op,
    dstT* dst, srcT srcV,
    const UINT* __restrict__ dstStride,
    UINT dstIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    dst[uiIdxSrc] = op.Do(srcV);
}

#if 0

template <class T>
__global__ void _CN_LAUNCH_BOUND _kernel_NaiveOneOperator(
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

/**
 * Only thread.x and block.x is used, so thread ID is simple
 * mutipliedlengths is calculated from "length"
 */
template <class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND _kernel_BlockCopy(
    dstT* dst,
    const UINT* __restrict__ dstStride,
    const UINT dstIndexStart,
    const srcT* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT srcIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxDst = _deviceThreadIdxToTensorIdxNaive(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    dst[uiIdxDst] = static_cast<dstT>(src[uiIdxSrc]);
}

template <class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND _kernel_BlockCopy_Small(
    dstT* dst,
    const UINT* __restrict__ dstStride,
    const UINT dstIndexStart,
    const srcT* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT srcIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(threadIdx.x + blockIdx.x * blockDim.x, mutipliedlengths, byIndexCount);
    const UINT uiIdxDst = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, dstStride, dstIndexStart, byIndexCount);
    const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);
    dst[uiIdxDst] = static_cast<dstT>(src[uiIdxSrc]);
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

#endif

#pragma endregion

template<class T>
template<class Operator>
void CNDeviceTensorCommonNaive<T>::OneOperator(
    const TOperator_D<Operator, T>& op,
    CNDeviceTensor<T>* dst,
    const UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount)
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 2;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetSmallDeviceBuffer(totalBufferSize);

    UINT* hostBuffer = (UINT*)appAlloca(dataSize);
    _memcpy_hd(deviceBuffer, dstStride, dataSize);
    __BuildMultiplyLength(deviceBuffer + dataSize);

    __KERNALCALNAIVE(_kernel_NaiveOneOperator,
        op,
        dst->m_pDeviceDataBuffer,
        (UINT*)deviceBuffer,
        dstIndexStart,
        (UINT*)(deviceBuffer + dataSize),
        byIndexCount
    );
}

#if 0

template<class dstT, class srcT>
void CNDeviceTensorCommonNaive::BlockCopy(
    CNDeviceTensor<dstT>* dst,
    const UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    const CNDeviceTensor<srcT>* __restrict__ src,
    const UINT srcIndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount)
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 3;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetTensorOpWorkingSpace()->GetSmallDeviceBuffer(totalBufferSize);
    _memcpy_hd(deviceBuffer, dstStride, dataSize);
    _memcpy_hd(deviceBuffer + dataSize, srcStride, dataSize);
    __BuildMultiplyLength(deviceBuffer + (dataSize << 1));

    __KERNALCALNAIVE(_kernel_BlockCopy,
        dst,
        (UINT*)deviceBuffer,
        dstIndexStart,
        src,
        (UINT*)(deviceBuffer + dataSize),
        srcIndexStart,
        (UINT*)(deviceBuffer + (dataSize << 1)),
        byIndexCount
    );
}

template <class T> __DLL_EXPORT
void FillMasked(T* src, const T& val,
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

template <class T> __DLL_EXPORT
void Transpose(T* dst,
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


#endif

#pragma region Implementation of Calculators

//template<class Operator>
//void CNHostTensor<_SComplex>::OneOperator(Operator op,
//    const CNDeviceTensorCommonNaive& calc,
//    const UINT dstIndexStart,
//    const UINT* __restrict__ dstStride,
//    const UINT* __restrict__ lengths,
//    BYTE byIndexCount)
//{
//    
//}

#pragma endregion

__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
