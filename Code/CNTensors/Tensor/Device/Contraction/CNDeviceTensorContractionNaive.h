//=============================================================================
// FILENAME : CNDeviceTensorContractionNaive.h
// 
// DESCRIPTION:
// 
// Meanings of three indexes
//
//
// REVISION:
//  [26/04/2022 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_CONTRACTION_NAIVE_H_
#define _CNDEVICETENSOR_CONTRACTION_NAIVE_H_

#define _CN_CONTRACTION_INDEX_COUNT_ONE_TIME 16

#define __IMPLEMENT_ContractionNaiveOneType(type, op) \
template class CNDeviceTensorContractionNaiveOneType<TOperator_##op<type, type>, type>;


#define __IMPLEMENT_ContractionNaiveTwoType(type1, type2) \
template class CNDeviceTensorContractionNaiveTwoType<type1, type2>;

__BEGIN_NAMESPACE

template<class Operator, class Tdst>
class __DLL_EXPORT CNDeviceTensorContractionNaiveOneType
{

public:

    CNDeviceTensorContractionNaiveOneType()
        : m_op()
    {

    }

    void SumAndProd(
        Tdst* pDstBuffer,
        const Tdst* __restrict__ pSrcBuffer,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount,
        UINT uiSumLength,
        UINT uiSumIndexStride);

    static inline UINT GetReduceDim(UINT uiLength)
    {
        UINT iRet = 0;
        while ((1U << iRet) < uiLength)
        {
            ++iRet;
        }
        return iRet;
    }

    /**
    * Note that, the tensor will be massed up
    */
    Tdst ReduceSumAndProdAll(Tdst* pBuffer, UINT uiSize);

    TOperator_S<Operator, Tdst, Tdst> m_op;
};

template<class Tdst, class Tsrc>
class __DLL_EXPORT CNDeviceTensorContractionNaiveTwoType
{
public:

    void Contraction(Tdst* pDstBuffer,
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
        UBOOL bConjugate) const;

    void Contraction(Tdst* pDstBuffer,
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
        UBOOL bConjugate) const;
};

class __DLL_EXPORT CNDeviceTensorContractionNaive : public TCNDeviceTensorContraction<CNDeviceTensorContractionNaive>
{
public:

    template<class Tdst>
    void Sum(Tdst* pDstBuffer,
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
        CNDeviceTensorContractionNaiveOneType<TOperator_Add<Tdst, Tdst>, Tdst>().SumAndProd(
            pDstBuffer, pSrcBuffer,
            dstIndexStart, dstStride, 
            srcIndexStart, srcStride, lengths, 
            byIndexCount, uiSumLength, uiSumIndexStride);
    }

    template<class Tdst>
    void Prod(Tdst* pDstBuffer,
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
        CNDeviceTensorContractionNaiveOneType<TOperator_Mul<Tdst, Tdst>, Tdst>().SumAndProd(
            pDstBuffer, pSrcBuffer,
            dstIndexStart, dstStride,
            srcIndexStart, srcStride, lengths,
            byIndexCount, uiSumLength, uiSumIndexStride);
    }

    template<class Tdst>
    Tdst ReduceSum(Tdst* pBuffer, UINT uiSize)
    {
        return CNDeviceTensorContractionNaiveOneType<TOperator_Add<Tdst, Tdst>, Tdst>().ReduceSumAndProdAll(pBuffer, uiSize);
    }

    template<class Tdst>
    Tdst ReduceProd(Tdst* pBuffer, UINT uiSize)
    {
        return CNDeviceTensorContractionNaiveOneType < TOperator_Mul<Tdst, Tdst>, Tdst > ().ReduceSumAndProdAll(pBuffer, uiSize);
    }

    template<class Tdst, class Tsrc>
    void Contraction(Tdst* pDstBuffer,
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
        UBOOL bConjugate)
    {
        CNDeviceTensorContractionNaiveTwoType<Tdst, Tsrc>().Contraction(
            pDstBuffer,
            pSrc1Buffer,
            pSrc2Buffer,
            dstIndexStart,
            dstStride,
            src1IndexStart,
            src2IndexStart,
            src1Stride,
            src2Stride,
            lengths,
            byIndexCount,
            byIndexCountLeft,
            uiSumLength,
            uiSumIndexStride1,
            uiSumIndexStride2,
            bConjugate
        );
    }

    template<class Tdst, class Tsrc>
    void Contraction(Tdst* pDstBuffer,
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
        UBOOL bConjugate)
    {
        CNDeviceTensorContractionNaiveTwoType<Tdst, Tsrc>().Contraction(
            pDstBuffer,
            pSrc1Buffer,
            pSrc2Buffer,
            dstIndexStart,
            dstStride,
            src1IndexStart,
            src2IndexStart,
            src1Stride,
            src2Stride,
            lengths,
            byIndexCount,
            byIndexCountLeft,
            sumLeftStride,
            sumRightStride,
            sumlengths,
            bySumIndexCount,
            bConjugate
        );
    }

};



__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_CONTRACTION_NAIVE_H_

//=============================================================================
// END OF FILE
//=============================================================================