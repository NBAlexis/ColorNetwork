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
__IMPLEMENT_COMMON_ONE(CNDeviceTensorCommonNaiveOneOperator)

#pragma region kernels

template <class Operator, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_TotalOneOp(TOperator_D<Operator, srcT> op, srcT* src)
{
    UINT uiTensorIdx = threadIdx.x + blockIdx.x * blockDim.x;
    op.Do(src[uiTensorIdx]);
}

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

#pragma endregion

template<class Operator, class T>
void CNDeviceTensorCommonNaiveOneOperator<Operator, T>::OneOperator(T* pBuffer, UINT uiTotalSize)
{
    UINT block = 1;
    UINT thread = 1;
    GetDecompose(BOUND_THREAD, uiTotalSize, block, thread);
    _kernel_TotalOneOp << <block, thread >> > (m_op, pBuffer);
}

/**
* dest[].Sin(dest)
* ...
*/
template<class Operator, class T>
void CNDeviceTensorCommonNaiveOneOperator<Operator, T>::OneOperator(
    T* pBuffer,
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
        m_op,
        pBuffer,
        (UINT*)deviceBuffer,
        dstIndexStart,
        (UINT*)(deviceBuffer + dataSize),
        byIndexCount
    );
}


__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
