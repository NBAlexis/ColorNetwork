//=============================================================================
// FILENAME : CNDeviceTensorContractionNaiveOneType.cu
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [26/04/2021 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

//This is the critical specialization
__OVER_ALL_TYPE_ONEA(__IMPLEMENT_ContractionNaiveOneType, Add)

__OVER_ALL_TYPE_ONEA(__IMPLEMENT_ContractionNaiveOneType, Mul)

#pragma region kernels

template <class Operator, class dstT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveSumOrProdct(
    TOperator_S<Operator, dstT, dstT> op,
    dstT* dst,
    const dstT* __restrict__ src,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT srcIndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount,
    UINT sumStride,
    UINT sumLength)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxDst = _deviceThreadIdxToTensorIdxNaive(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    dst[uiIdxDst] = src[uiIdxSrc];
    #pragma unroll
    for (UINT i = 1; i < sumLength; ++i)
    {
        op.Do(dst[uiIdxDst], src[uiIdxSrc + i * sumStride]);
    }
}

template <class Operator, class dstT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveSumOrProdct_Small(
    TOperator_S<Operator, dstT, dstT> op,
    dstT* dst,
    const dstT* __restrict__ src,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT srcIndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount,
    UINT sumStride,
    UINT sumLength)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
    const UINT uiIdxDst = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, dstStride, dstIndexStart, byIndexCount);
    const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);
    dst[uiIdxDst] = src[uiIdxSrc];
    #pragma unroll
    for (UINT i = 1; i < sumLength; ++i)
    {
        op.Do(dst[uiIdxDst], src[uiIdxSrc + i * sumStride]);
    }
}

template <class Operator, class dstT>
__global__ void _CN_LAUNCH_BOUND
_kernelReduce(TOperator_S<Operator, dstT, dstT> op, dstT* arr, UINT uiJump, UINT uiMax)
{
    UINT uiIdFrom = (threadIdx.x + blockIdx.x * blockDim.x) * (uiJump << 1) + uiJump;
    if (uiIdFrom < uiMax)
    {
        op.Do(arr[uiIdFrom - uiJump], arr[uiIdFrom]);
    }
}

#pragma endregion

template<class Operator, class Tdst>
void CNDeviceTensorContractionNaiveOneType<Operator, Tdst>::SumAndProd(
    Tdst* pDstBuffer,
    const Tdst* __restrict__ pSrcBuffer,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT srcIndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount,
    UINT uiSumLength,
    UINT uiSumIndexStride)
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 3;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetSmallDeviceBuffer(totalBufferSize);

    _memcpy_hd(deviceBuffer, dstStride, dataSize);
    _memcpy_hd(deviceBuffer + dataSize, srcStride, dataSize);
    UINT* hostBuffer = (UINT*)appAlloca(dataSize);
    __BuildMultiplyLength(deviceBuffer + (dataSize << 1));

    __KERNALCALNAIVE(_kernel_NaiveSumOrProdct,
        m_op,
        pDstBuffer,
        pSrcBuffer,
        dstIndexStart,
        (UINT*)deviceBuffer,
        srcIndexStart,
        (UINT*)(deviceBuffer + dataSize),
        (UINT*)(deviceBuffer + (dataSize << 1)),
        byIndexCount,
        uiSumIndexStride,
        uiSumLength
    );

}

template<class Operator, class Tdst>
Tdst CNDeviceTensorContractionNaiveOneType<Operator, Tdst>::ReduceSumAndProdAll(Tdst* pBuffer, UINT uiSize)
{
    const UINT iRequiredDim = (uiSize + 1) >> 1;
    const UINT iPower = GetReduceDim(iRequiredDim);
    for (UINT i = 0; i <= iPower; ++i)
    {
        UINT iJump = 1 << i;
        UINT iThreadNeeded = 1 << (iPower - i);
        UINT iBlock = iThreadNeeded > BOUND_THREAD ? (iThreadNeeded / BOUND_THREAD) : 1;
        UINT iThread = iThreadNeeded > BOUND_THREAD ? BOUND_THREAD : iThreadNeeded;
        _kernelReduce << <iBlock, iThread >> > (m_op, pBuffer, iJump, uiSize);
    }
    Tdst result[1];
    checkCudaErrors(cudaMemcpy(result, pBuffer, sizeof(Tdst), cudaMemcpyDeviceToHost));
    return result[0];
}


__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
