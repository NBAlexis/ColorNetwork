//=============================================================================
// FILENAME : CNDeviceTensorContractionNaiveTwoType.cu
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
__OVER_ALL_TYPE_TWO(__IMPLEMENT_ContractionNaiveTwoType)

#pragma region kernels

template <class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveContract(
    dstT* dst,
    const dstT* __restrict__ src1,
    const srcT* __restrict__ src2,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT src1IndexStart,
    UINT src2IndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount,
    BYTE byIndexLeft,
    UINT sumStride1,
    UINT sumStride2,
    UINT sumLength,
    UBOOL bConjugate)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxDst = _deviceThreadIdxToTensorIdxNaive(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    UINT uiIdxSrc1 = src1IndexStart;
    UINT uiIdxSrc2 = src2IndexStart;
    _deviceThreadIdxToTensorIdxNaiveLR(uiIdxSrc1, uiIdxSrc2, byIndexLeft, srcStride, mutipliedlengths, uiIdx, byIndexCount);
    dst[uiIdxDst] = _Mul(bConjugate ? _Conj(src1[uiIdxSrc1]) : src1[uiIdxSrc1], src2[uiIdxSrc2]);
    #pragma unroll
    for (UINT i = 1; i < sumLength; ++i)
    {
        dst[uiIdxDst] = _Add(dst[uiIdxDst], _Mul(bConjugate ? _Conj(src1[uiIdxSrc1 + i * sumStride1]) : src1[uiIdxSrc1 + i * sumStride1], src2[uiIdxSrc2 + i * sumStride2]));
    }
}

template <class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveContract_Small(
    dstT* dst,
    const dstT* __restrict__ src1,
    const srcT* __restrict__ src2,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT src1IndexStart,
    UINT src2IndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount,
    BYTE byIndexLeft,
    UINT sumStride1,
    UINT sumStride2,
    UINT sumLength,
    UBOOL bConjugate)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
    const UINT uiIdxDst = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, dstStride, dstIndexStart, byIndexCount);
    UINT uiIdxSrc1 = src1IndexStart;
    UINT uiIdxSrc2 = src2IndexStart;
    _deviceWorkIndexToTensorIndexNaiveLR(uiIdxSrc1, uiIdxSrc2, byIndexLeft, idx.m_Idx, srcStride, byIndexCount);
    dst[uiIdxDst] = _Mul(bConjugate ? _Conj(src1[uiIdxSrc1]) : src1[uiIdxSrc1], src2[uiIdxSrc2]);
    #pragma unroll
    for (UINT i = 1; i < sumLength; ++i)
    {
        dst[uiIdxDst] = _Add(dst[uiIdxDst], _Mul(bConjugate ? _Conj(src1[uiIdxSrc1 + i * sumStride1]) : src1[uiIdxSrc1 + i * sumStride1], src2[uiIdxSrc2 + i * sumStride2]));
    }
}

template <class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveContractM(
    dstT* dst,
    const dstT* __restrict__ src1,
    const srcT* __restrict__ src2,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT src1IndexStart,
    UINT src2IndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount,
    BYTE byIndexLeft,
    const UINT* __restrict__ sumStride1,
    const UINT* __restrict__ sumStride2,
    const UINT* __restrict__ sumLength,
    BYTE bySumIndexCount,
    UBOOL bConjugate)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiIdxDst = _deviceThreadIdxToTensorIdxNaive(dstStride, dstIndexStart, mutipliedlengths, uiIdx, byIndexCount);
    UINT uiIdxSrc1 = src1IndexStart;
    UINT uiIdxSrc2 = src2IndexStart;
    _deviceThreadIdxToTensorIdxNaiveLR(uiIdxSrc1, uiIdxSrc2, byIndexLeft, srcStride, mutipliedlengths, uiIdx, byIndexCount);
    dst[uiIdxDst] = _deviceNestedSum(src1, src2, uiIdxSrc1, uiIdxSrc2, sumStride1, sumStride2, sumLength, 0, bySumIndexCount, bConjugate);
}

template <class dstT, class srcT>
__global__ void _CN_LAUNCH_BOUND
_kernel_NaiveContractM_Small(
    dstT* dst,
    const dstT* __restrict__ src1,
    const srcT* __restrict__ src2,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT src1IndexStart,
    UINT src2IndexStart,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ mutipliedlengths,
    BYTE byIndexCount,
    BYTE byIndexLeft,
    const UINT* __restrict__ sumStride1,
    const UINT* __restrict__ sumStride2,
    const UINT* __restrict__ sumLength,
    BYTE bySumIndexCount,
    UBOOL bConjugate)
{
    const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
    const UINT uiIdxDst = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, dstStride, dstIndexStart, byIndexCount);
    UINT uiIdxSrc1 = src1IndexStart;
    UINT uiIdxSrc2 = src2IndexStart;
    _deviceWorkIndexToTensorIndexNaiveLR(uiIdxSrc1, uiIdxSrc2, byIndexLeft, idx.m_Idx, srcStride, byIndexCount);
    dst[uiIdxDst] = _deviceNestedSum(src1, src2, uiIdxSrc1, uiIdxSrc2, sumStride1, sumStride2, sumLength, 0, bySumIndexCount, bConjugate);
}

#pragma endregion

template<class Tdst, class Tsrc>
void CNDeviceTensorContractionNaiveTwoType<Tdst, Tsrc>::Contraction(
    Tdst* pDstBuffer,
    const Tdst* __restrict__ pSrc1Buffer,
    const Tsrc* __restrict__ pSrc2Buffer,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT src1IndexStart,
    UINT src2IndexStart,
    const UINT* __restrict__ src1Stride,
    const UINT* __restrict__ src2Stride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount,
    BYTE byIndexCountLeft,
    UINT uiSumLength,
    UINT uiSumIndexStride1,
    UINT uiSumIndexStride2,
    UBOOL bConjugate) const
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT totalBufferSize = dataSize * 3;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetSmallDeviceBuffer(totalBufferSize);

    UINT* hostBuffer = (UINT*)appAlloca(dataSize);
    _memcpy_hd(deviceBuffer, dstStride, dataSize);
    for (BYTE byOrder = 0; byOrder < byIndexCount; ++byOrder)
    {
        if (byOrder < byIndexCountLeft)
        {
            hostBuffer[byOrder] = src1Stride[byOrder];
        }
        else
        {
            hostBuffer[byOrder] = src2Stride[byOrder - byIndexCountLeft];
        }
    }
    _memcpy_hd(deviceBuffer + dataSize, hostBuffer, dataSize);
    __BuildMultiplyLength(deviceBuffer + (dataSize << 1));

    __KERNALCALNAIVE(_kernel_NaiveContract,
        pDstBuffer,
        pSrc1Buffer,
        pSrc2Buffer,
        dstIndexStart,
        (UINT*)deviceBuffer,
        src1IndexStart,
        src2IndexStart,
        (UINT*)(deviceBuffer + dataSize),
        (UINT*)(deviceBuffer + (dataSize << 1)),
        byIndexCount,
        byIndexCountLeft,
        uiSumIndexStride1,
        uiSumIndexStride2,
        uiSumLength,
        bConjugate
    );
}

template<class Tdst, class Tsrc>
void CNDeviceTensorContractionNaiveTwoType<Tdst, Tsrc>::Contraction(Tdst* pDstBuffer,
    const Tdst* __restrict__ pSrc1Buffer,
    const Tsrc* __restrict__ pSrc2Buffer,
    UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    UINT src1IndexStart,
    UINT src2IndexStart,
    const UINT* __restrict__ src1Stride,
    const UINT* __restrict__ src2Stride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount,
    BYTE byIndexCountLeft,
    const UINT* __restrict__ sumLeftStride,
    const UINT* __restrict__ sumRightStride,
    const UINT* __restrict__ sumlengths,
    BYTE bySumIndexCount,
    UBOOL bConjugate) const
{
    const UINT dataSize = sizeof(UINT) * byIndexCount;
    const UINT dataSizeSum = sizeof(UINT) * bySumIndexCount;
    const UINT totalBufferSize = dataSize * 3 + dataSizeSum * 3;
    UINT uiBlock, uiThread;
    SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

    BYTE* deviceBuffer = appGetSmallDeviceBuffer(totalBufferSize);

    UINT* hostBuffer = (UINT*)appAlloca(dataSize);
    _memcpy_hd(deviceBuffer, dstStride, dataSize);
    for (BYTE byOrder = 0; byOrder < byIndexCount; ++byOrder)
    {
        if (byOrder < byIndexCountLeft)
        {
            hostBuffer[byOrder] = src1Stride[byOrder];
        }
        else
        {
            hostBuffer[byOrder] = src2Stride[byOrder - byIndexCountLeft];
        }
    }
    _memcpy_hd(deviceBuffer + dataSize, hostBuffer, dataSize);
    __BuildMultiplyLength(deviceBuffer + (dataSize << 1));

    _memcpy_hd(deviceBuffer + 3 * dataSize, sumLeftStride, dataSizeSum);
    _memcpy_hd(deviceBuffer + 3 * dataSize + dataSizeSum, sumRightStride, dataSizeSum);
    _memcpy_hd(deviceBuffer + 3 * dataSize + (dataSizeSum << 1), sumlengths, dataSizeSum);

    __KERNALCALNAIVE(_kernel_NaiveContractM,
        pDstBuffer,
        pSrc1Buffer,
        pSrc2Buffer,
        dstIndexStart,
        (UINT*)deviceBuffer,
        src1IndexStart,
        src2IndexStart,
        (UINT*)(deviceBuffer + dataSize),
        (UINT*)(deviceBuffer + (dataSize << 1)),
        byIndexCount,
        byIndexCountLeft,
        deviceBuffer + 3 * dataSize,
        deviceBuffer + 3 * dataSize + dataSizeSum,
        deviceBuffer + 3 * dataSize + (dataSizeSum << 1),
        bySumIndexCount,
        bConjugate
    );
}



__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
