//=============================================================================
// FILENAME : CNDeviceTensorContraction.h
// 
// DESCRIPTION:
// This is an intereface
//
// REVISION[d-m-y]:
//  [26/04/2022 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_CONTRACTION_H_
#define _CNDEVICETENSOR_CONTRACTION_H_

//==========================================================

__BEGIN_NAMESPACE

template<class Calc>
class __DLL_EXPORT TCNDeviceTensorContraction
{
public:

    virtual ~TCNDeviceTensorContraction()
    {

    }

    /**
    * b[i1,i2,i4,i5] = Sum _ i3 a[i1,i2,i3,i4,i5]
    * 
    * more specifically:
    * b[i1*stride1 + i2*stride2 + i4*stride4 + i5*stride5] = sum _k=0^uiSumLength-1
    * a[i1*stride1 + i2*stride2 + i4*stride4 + i5*stride5 + uiSumIndexStart + k*uiSumIndexStride]
    * note that the uiSumIndexStart can be absorbed into the srcIndexStart
    * 
    * Self contraction can be realized similarly
    * b[i1,i2,i4,i5] = Sum _ {i3,i4} a[i1,i2,i3,i4,i5,i6] delta(i3, i4)
    * which is
    * b[i1*stride1 + i2*stride2 + i4*stride4 + i5*stride5] = sum _k=0^uiSumLength-1
    * a[i1*stride1 + i2*stride2 + i4*stride4 + i5*stride5 + uiSumIndexStart + k*(uiSumIndexStride1 + uiSumIndexStride2)]
    */
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
        ((Calc*)this)->Sum(pDstBuffer, pSrcBuffer, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount, uiSumLength, uiSumIndexStride);
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
        ((Calc*)this)->Prod(pDstBuffer, pSrcBuffer, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, uiSumLength, uiSumIndexStride);
    }

    /**
    * Note that reduce will mass up the tensor buffer
    * If you need a blocked sum, copy the block out before sum...(since the tensor will mass up)
    */
    template<class Tdst>
    Tdst ReduceSum(Tdst* pBuffer, UINT uiSize)
    {
        return ((Calc*)this)->ReduceSum(pBuffer, uiSize);
    }

    template<class Tdst>
    Tdst ReduceProd(Tdst* pBuffer, UINT uiSize)
    {
        return ((Calc*)this)->ReduceProd(pBuffer, uiSize);
    }

    /**
    * c[i1,i2,i4,i5] = Sum _ {i3} a[i1,i2,i3] b[i3,i4,i5]
    *
    * more specifically:
    * c[dstIndexStart + i1*stride1 + i2*stride2 + i4*stride4 + i5*stride5] = sum _k=0^uiSumLength-1
    * a[i1*stride1 + i2*stride2 + src1IndexStart + k*uiSumIndexStride1] x
    * b[i4*stride4 + i5*stride5 + src2IndexStart + k*uiSumIndexStride2]
    * 
    * note that, byIndexCount = byIndexCountLeft + byIndexCountRight
    */
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
        UINT uiSumIndexStride2)
    {
        ((Calc*)this)->Contraction(
            pDstBuffer, pSrc1Buffer, pSrc2Buffer, 
            dstIndexStart, dstStride, src1IndexStart, src1Stride, src2IndexStart, src2Stride, 
            lengths, byIndexCount, byIndexCountLeft, uiSumLength, uiSumIndexStride1, uiSumIndexStride2);
    }

    /**
    * c[i1,i2,i5,i6] = Sum _ {i3,i4} a[i1,i2,i3,i4] b[i3,i4,i5,i6]
    *
    * more specifically:
    * c[dstIndexStart + i1*stride1 + i2*stride2 + i5*stride5 + i6*stride6] = sum _ki=0^sumlengths[i]-1
    * a[src1IndexStart + i1*stride1 + i2*stride2 + ki*sumLeftStride[i]] x
    * b[src2IndexStart + i5*stride5 + i6*stride6 + ki*sumRightStride[i]]
    *
    * note that, byIndexCount = byIndexCountLeft + byIndexCountRight
    */
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
        BYTE bySumIndexCount)
    {
        ((Calc*)this)->Contraction(
            pDstBuffer, pSrc1Buffer, pSrc2Buffer,
            dstIndexStart, dstStride, src1IndexStart, src1Stride, src2IndexStart, src2Stride,
            lengths, byIndexCount, byIndexCountLeft, sumLeftStride, sumRightStride, sumlengths, bySumIndexCount);
    }
};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_CONTRACTION_H_

//=============================================================================
// END OF FILE
//=============================================================================