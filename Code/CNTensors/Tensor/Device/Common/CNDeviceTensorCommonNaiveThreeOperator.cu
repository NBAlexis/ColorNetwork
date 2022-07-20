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
__OVER_ALL_TYPE_TWOA(__IMPLEMENT_COMMON_THREEAPXY, CNDeviceTensorCommonThreeOperatorNaive)

#pragma region kernels

template <class Operator1, class Operator2, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND _kernel_NaiveAxpy(
    TOperator_S<Operator1, dstT, dstT> op1,
    TOperator_S<Operator2, dstT, srcT> op2,
    dstT* dst,
    const UINT* __restrict__ dstStride,
    const UINT dstIndexStart,
    dstT srcA,
    const srcT* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT srcIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxDst = _deviceThreadIdxToTensorIdxNaive(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    op1.Do(dst[uiIdxDst], op2.Dor(srcA, src[uiIdxSrc]));
}

template <class Operator1, class Operator2, class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND _kernel_NaiveAxpy_Small(
    TOperator_S<Operator1, dstT, dstT> op1,
    TOperator_S<Operator2, dstT, srcT> op2,
    dstT* dst,
    const UINT* __restrict__ dstStride,
    const UINT dstIndexStart,
    dstT srcA,
    const srcT* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT srcIndexStart,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount)
{
    const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(threadIdx.x + blockIdx.x * blockDim.x, mutipliedlengths, byIndexCount);
    const UINT uiIdxDst = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, dstStride, dstIndexStart, byIndexCount);
    const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);
    op1.Do(dst[uiIdxDst], op2.Dor(srcA, src[uiIdxSrc]));
}

#pragma endregion

/**
* dest[] = dest[] + src1 * src2
*/
template<class Operator1, class Operator2, class Tdst, class Tsrc>
void CNDeviceTensorCommonThreeOperatorNaive<Operator1, Operator2, Tdst, Tsrc>::ThreeOperatorTensor(
    Tdst* pBuffer,
    const Tdst& v,
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

    __KERNALCALNAIVE(_kernel_NaiveAxpy,
        m_op1,
        m_op2,
        pBuffer,
        (UINT*)deviceBuffer,
        dstIndexStart,
        v,
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
