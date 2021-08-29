//=============================================================================
// FILENAME : CNDeviceTensorCommon.h
// 
// DESCRIPTION:
// This is an intereface
//
// REVISION:
//  [18/06/2021 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_COMMON_H_
#define _CNDEVICETENSOR_COMMON_H_

__BEGIN_NAMESPACE

#if 0

/**
 * In device interfaces, we do NOT check whether things can be done
 */
class CNAPI CNDeviceTensorCommon
{
public:
    virtual ~CNDeviceTensorCommon() {}

    /**
     * dst[..., i0 + i, ..., j0 + j] = src[..., i0' + i, ..., j0' + j]
     * The stride is for things like this:
     *    dst[i0 + i, j0] = src[i0' + i]
     * or dst[i0, j0 + i] = src[i0' + i]
     *
     * The i0, j0, ..., and i0', j0' are encoded as
     * flatten indices dstIndexStart and srcIndexStart
     * 
     * len(dstStride) = len(srcStride) = len(lengths) = byIndexCount
     * this is in fact
     * for i, j, k in lengths
     *    dst[dstIndexStart + i * dststride1 + j * dststrde2 + k * dststrde3]
     *  = src[srcIndexStart + i * srcstride1 + j * srcstrde2 + k * srcstrde3]
     *
     * dstStride, srcStride, lengths are host buffers
     */
    template<class dstT, class srcT>
    void BlockCopy(
        CNDeviceTensor<dstT>* dst,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const CNDeviceTensor<srcT>* __restrict__ src,
        const UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths, 
        BYTE byIndexCount)
    {
        appCrucial(_T("BlockCopyMasked not implemented!\n"));
    }

    /**
     * if mask[...] != 0, then T[..., i0 + i, ..., j0 + j] = val
     */
    template<class srcT, class valT>
    void FillMasked(CNDeviceTensor<srcT>* src, const valT& val,
        const UINT* __restrict__ srcIndexStart,
        const BYTE* __restrict__ mask,
        const UINT* __restrict__ maskStride,
        const UINT* __restrict__ maskIndexStart,
        const UINT* __restrict__ lengths, BYTE byIndexCount)
    {
        appCrucial(_T("FillMasked not implemented!\n"));
    }

    /**
     * T[..., i0 + i, ..., j0 + j] = val
     */
    template<class srcT, class valT>
    inline void Fill(CNDeviceTensor<srcT>* src, const valT& val,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ srcStart,
        const UINT* __restrict__ lengths, BYTE byIndexCount)
    {
        FillMasked(src, val, srcStride, srcStart, NULL, NULL, NULL, lengths, byIndexCount);
    }

    /**
     * T[..., i, ..., j] = val
     */
    template<class srcT, class valT>
    inline void Fill(CNDeviceTensor<srcT>* src, const valT& val,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths, BYTE byIndexCount)
    {
        Fill(src, val, srcStride, NULL, lengths, byIndexCount);
    }

    template <class srcT>
    void RandomMasked(CNDeviceTensor<srcT>* src,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ srcStart,
        const UINT* __restrict__ srcEnd,
        const BYTE* __restrict__ mask,
        const UINT* __restrict__ maskStride,
        const UINT* __restrict__ maskIndexStart,
        const UINT* __restrict__ lengths, BYTE byIndexCount)
    {
        appCrucial(_T("RandomMasked not implemented!\n"));
    }

    template <class srcT>
    inline void Random(CNDeviceTensor<srcT>* src,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ srcStart,
        const UINT* __restrict__ lengths, BYTE byIndexCount)
    {
        
    }

    /**
     * T[x1, x2, x3, ..., y1, y2, y3, ...] = T[ x2, y1, x1, x3, y4, ...]
     * order Stride and Dims as stride(dim) of x1, x2, x3, ..., y1, y2, y3.
     * Note: transpose must copy
     */
    template<class dstT, class srcT>
    void Transpose(CNDeviceTensor<dstT>* dst,
        const CNDeviceTensor<srcT>* __restrict__ src,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ srcDim,
        BYTE byIndexCount)
    {
        appCrucial(_T("Transpose not implemented!\n"));
    }

    /**
     * M[x, y] = T[ x2, y1, x1, x3, y4, ...]
     * order Stride and Dims as stride(dim) of x1, x2, x3, ..., y1, y2, y3, Then everything automatically happen
     *
     * Explanation:
     * threadIdx = x * dimy + y
     * x1 = threadIdx / x2*x3*...*y1*y2*y3
     * x2 = (threadIdx % x2*x3*...*y1*y2*y3) % (x3*...*y1*y2*y3)
     * so the first thing is to make sure multiply lengthes are ordered.
     * TensorIdx = x1 * stride1 + x2 * stride2 + ...
     * so the next thing is to make sure the strides are ordered.
     */
    template<class dstT, class srcT>
    inline void ToMatrix(CNDeviceTensor<dstT>* dst,
        const CNDeviceTensor<srcT>* __restrict__ src,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ srcDim,
        BYTE byIndexCount)
    {
        Transpose(dst, src, srcStride, srcDim, byIndexCount);
    }

    template <class T> 
    void DebugPrint(
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

    template <class T> 
    void DebugPrint(
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
};

/**
 * This is called CRTP, see:
 * https://stackoverflow.com/questions/2354210/can-a-class-member-function-template-be-virtual#
 *
 * To implement a calculator, call
 * class Calculator : public TCNDeviceTensorCommon<Calculator>
 *
 * 
 *
 */
template<class Calculator>
class __DLL_EXPORT TCNDeviceTensorCommon : public CNDeviceTensorCommon
{
public:

    template<class dstT, class srcT>
    void BlockCopy(
        CNDeviceTensor<dstT>* dst,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const CNDeviceTensor<srcT>* __restrict__ src,
        const UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        ((Calculator*)this)->BlockCopy(dst, dstIndexStart, dstStride, src, srcIndexStart, srcStride, lengths, byIndexCount);
    }

    template<class srcT, class valT>
    void FillMasked(CNDeviceTensor<srcT>* src, const valT& val,
        const UINT* __restrict__ srcIndexStart,
        const UINT* __restrict__ lengths, BYTE byIndexCount)
    {
        ((Calculator*)this)->FillMasked(src, val, srcIndexStart, mask, maskStride, maskIndexStart, lengths, byIndexCount);
    }

    template <class srcT>
    void RandomMasked(CNDeviceTensor<srcT>* src,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ srcStart,
        const UINT* __restrict__ srcEnd,
        const BYTE* __restrict__ mask,
        const UINT* __restrict__ maskStride,
        const UINT* __restrict__ maskIndexStart,
        const UINT* __restrict__ lengths, BYTE byIndexCount)
    {
        appCrucial(_T("RandomMasked not implemented!\n"));
    }


    /**
     * T[x1, x2, x3, ..., y1, y2, y3, ...] = T[ x2, y1, x1, x3, y4, ...]
     * order Stride and Dims as stride(dim) of x1, x2, x3, ..., y1, y2, y3.
     * Note: transpose must copy
     */
    template<class dstT, class srcT>
    void Transpose(CNDeviceTensor<dstT>* dst,
        const CNDeviceTensor<srcT>* __restrict__ src,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ srcDim,
        BYTE byIndexCount)
    {
        appCrucial(_T("Transpose not implemented!\n"));
    }
};

#endif

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_H_

//=============================================================================
// END OF FILE
//=============================================================================