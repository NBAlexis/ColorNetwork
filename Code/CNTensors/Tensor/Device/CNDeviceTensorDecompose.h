//=============================================================================
// FILENAME : CNDeviceTensorDecompose.h
// 
// DESCRIPTION:
// This is an intereface
//
// REVISION[d-m-y]:
//  [21/05/2022 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_DECOMPOSE_H_
#define _CNDEVICETENSOR_DECOMPOSE_H_

//==========================================================

__BEGIN_NAMESPACE

#if 0
template<class Calc>
class __DLL_EXPORT TCNDeviceTensorDecompose
{
public:

    virtual ~TCNDeviceTensorDecompose()
    {

    }

    /**
    * compute src = QR
    * then, set
    * src = R
    * dst = Q
    * the matrix M[ij] = tensor[i * stride + j + indexstart]
    */
    template<class Tdst, class Tsrc>
    void QRFactorization(
        Tdst* pDstBuffer, 
        Tsrc* pSrcBuffer,
        UINT srcIndexStart,
        UINT srcStride,
        UINT dstIndexStart,
        UINT dstStride,
        UINT idxLength1,
        UINT idxLength2)
    {
        ((Calc*)this)->Sum(pDstBuffer, pSrcBuffer, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount, uiSumLength, uiSumIndexStride);
    }


};
#endif

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_DECOMPOSE_H_

//=============================================================================
// END OF FILE
//=============================================================================