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

//This is the critical specialization
__OVER_ALL_TYPE_TWOAB(__IMPLEMENT_COMMON_TWO_SET, Set, CNDeviceTensorCommonTwoOperatorNaive)

__IMPLEMENT_COMMON_TWO(CNDeviceTensorCommonTwoOperatorNaive)

#pragma region kernels

template <class Operator, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_TotalTwoOperator_Value(
    TOperator_S<Operator, dstT, srcT> op,
    dstT* dst, srcT srcV)
{
    UINT uiTensorIdx = threadIdx.x + blockIdx.x * blockDim.x;
    op.Do(dst[uiTensorIdx], srcV);
}

template <class Operator, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveTwoOperator_Value(
    TOperator_S<Operator, dstT, srcT> op,
    dstT* dst, srcT srcV,
    const UINT* __restrict__ dstStride,
    UINT dstIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    op.Do(dst[uiIdxSrc], srcV);
}

template <class Operator, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveTwoOperator_Value_Small(
    TOperator_S<Operator, dstT, srcT> op,
    dstT* dst, srcT srcV,
    const UINT* __restrict__ dstStride,
    UINT dstIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
    const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, dstStride, dstIndexStart, byIndexCount);
    op.Do(dst[uiIdxSrc], srcV);
}

template <class Operator, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND _kernel_NaiveTwoOperator_Tensor(
    TOperator_S<Operator, dstT, srcT> op,
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
    op.Do(dst[uiIdxDst], src[uiIdxSrc]);
}

template <class Operator, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND _kernel_NaiveTwoOperator_Tensor_Small(
    TOperator_S<Operator, dstT, srcT> op,
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
    op.Do(dst[uiIdxDst], src[uiIdxSrc]);
}

#pragma endregion

template<class Operator, class Tdst, class Tsrc>
void CNDeviceTensorCommonTwoOperatorNaive<Operator, Tdst, Tsrc>::TwoOperatorValueTotal(
    Tdst* pBuffer,
    const Tsrc& v,
    UINT uiTotalSize)
{
    UINT block = 1;
    UINT thread = 1;
    GetDecompose(BOUND_THREAD, uiTotalSize, block, thread);
    _kernel_TotalTwoOperator_Value << <block, thread >> > (m_op, pBuffer, v);
}

/**
* dest[] = src...
* dest[].Add(src)...
*/
template<class Operator, class Tdst, class Tsrc>
void CNDeviceTensorCommonTwoOperatorNaive<Operator, Tdst, Tsrc>::TwoOperatorValue(
    Tdst* pBuffer,
    const Tsrc& v,
    UINT dstIndexStart,
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

    __KERNALCALNAIVE(_kernel_NaiveTwoOperator_Value,
        m_op,
        pBuffer,
        v,
        (UINT*)deviceBuffer,
        dstIndexStart,
        (UINT*)(deviceBuffer + dataSize),
        byIndexCount
    );
}

/**
* dest[] = src[]
* dest[].Add(src[])
* ...
*/
template<class Operator, class Tdst, class Tsrc>
void CNDeviceTensorCommonTwoOperatorNaive<Operator, Tdst, Tsrc>::TwoOperatorTensor(
    Tdst* pBuffer,
    const Tsrc* __restrict__ src,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT srcIndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount)
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 3;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetSmallDeviceBuffer(totalBufferSize);
    UINT* hostBuffer = (UINT*)appAlloca(dataSize);
    _memcpy_hd(deviceBuffer, dstStride, dataSize);
    _memcpy_hd(deviceBuffer + dataSize, srcStride, dataSize);
    __BuildMultiplyLength(deviceBuffer + (dataSize << 1));

    __KERNALCALNAIVE(_kernel_NaiveTwoOperator_Tensor,
        m_op,
        pBuffer,
        (UINT*)deviceBuffer,
        dstIndexStart,
        src,
        (UINT*)(deviceBuffer + dataSize),
        srcIndexStart,
        (UINT*)(deviceBuffer + (dataSize << 1)),
        byIndexCount
    );
}


__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
