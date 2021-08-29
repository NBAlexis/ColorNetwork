//=============================================================================
// FILENAME : CNDeviceTensorCommonNaive.h
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [19/06/2021 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_COMMON_NAIVE_H_
#define _CNDEVICETENSOR_COMMON_NAIVE_H_

#define __KERNALCALNAIVE(funcName, ...) \
    if (byIndexCount > _kSmallOrder) \
    { \
        funcName << <uiBlock, uiThread >> > (__VA_ARGS__); \
    } \
    else \
    { \
        funcName##_Small << <uiBlock, uiThread >> > (__VA_ARGS__); \
    }


#define __BuildMultiplyLength(ptr) \
    UINT* mul_length = appGetTensorOpWorkingSpace()->GetMultiplyLengthBuffer(); \
    mul_length[byIndexCount - 1] = 1; \
    for (INT i = byIndexCount - 2; i >= 0; --i) /* do not use BYTE here*/ \
    { \
        mul_length[i] = mul_length[i + 1] * lengths[i + 1]; \
    } \
    _memcpy_hd(ptr, mul_length, dataSize);

__BEGIN_NAMESPACE

#pragma region Index mapping

/**
 * In the case of TN, typically we have a tensor smaller than this
 */
constexpr BYTE _kSmallOrder = 8;
constexpr WORD _kSmallMaxDim = 256;
typedef WORD smallIdx;

/**
 * We only use the block.x and thread.x for simplicity.
 * blocks * threads = \prod_i lengths[i], with threads < uiMaxThread
 */
static inline void SimpleThreadDecompose(
    const UINT* __restrict__ lengths,
    BYTE byIndexCount,
    UINT& blocks,
    UINT& threads,
    UINT uiMaxThread = BOUND_THREAD)
{
    blocks = 1;
    threads = 1;
    for (BYTE i = 0; i < byIndexCount; ++i)
    {
        if (lengths[i] * threads < uiMaxThread)
        {
            threads = threads * lengths[i];
        }
        else
        {
            blocks = blocks * lengths[i];
        }
    }
}

/**
 * Map the linear index of thread to linear index of tensor
 * multpliedLengths[i] = \prod lengths[j>i]
 * (1) Calculate (a1, a2, ...) from uiIdx with multpliedLengths
 * (2) Calcuate linear index of T[a1+start1, a2+start2, ...] with strides
 * Take index [1,2,3,4] for example, multpliedLengths = {1000, 100, 10}
 *
 * uiIdx is the index of thread
 * return linear index of tensor
 */
__device__ static __inline__ UINT _deviceThreadIdxToTensorIdxNaive(
    const UINT* __restrict__ strides,
    const UINT start,
    const UINT* __restrict__ multpliedLengths,
    UINT uiIdx, BYTE byOrder)
{
    UINT uiRet = start;
    for (UINT i = 0; i < byOrder; ++i)
    {
        /**
         * Take 1234 for example
         * when i = 0,      1 = 1234 / 1000
         * when i = 1,2,    2 = (1234 % 1000) / 100,  3 = (1234 % 100) / 10
         * when i = 3,      4 = 1234 % 100
         */
        const UINT idxForThisOrder = (0 == i)
            ? (uiIdx / multpliedLengths[i])
            : ((byOrder == i + 1)
                ? (uiIdx % multpliedLengths[i - 1])
                : ((uiIdx % multpliedLengths[i - 1]) / multpliedLengths[i]));

        uiRet = uiRet + idxForThisOrder * strides[i];
    }
    return uiRet;
}



/**
 * a 128-bit index
 * I think use array is better than bit as long as it is properly aligned
 * smallIdx is unsigned short
 */
struct SWorkingIndex
{
    smallIdx m_Idx[_kSmallOrder];
};

/**
* map thread index to working indice
*/
__device__ static __inline__ SWorkingIndex _deviceThreadIndexToWorkIndexNavie(UINT uiIdx,
    const UINT* __restrict__ multpliedLengths, BYTE byOrder)
{
    SWorkingIndex ret;

    //#pragma unroll
    //for (BYTE i = 0; i < _kSmallOrder; ++i)
    //{
    //    if (i < byOrder)
    //    {
    //        ret.m_Idx[i] = static_cast<smallIdx>((0 == i) ? (uiIdx / multpliedLengths[i])
    //            : ((byOrder == i + 1)
    //                ? (uiIdx % multpliedLengths[i - 1])
    //                : ((uiIdx % multpliedLengths[i - 1]) / multpliedLengths[i])));
    //    }
    //    else
    //    {
    //        ret.m_Idx[i] = 0;
    //    }
    //}
    //unroll in .h file is unknown pragma

    #define ___addidx(i) \
    if (i < byOrder) \
    { \
        ret.m_Idx[i] = static_cast<smallIdx>((0 == i) ? (uiIdx / multpliedLengths[i]) \
            : ((byOrder == i + 1) \
                ? (uiIdx % multpliedLengths[i - 1]) \
                : ((uiIdx % multpliedLengths[i - 1]) / multpliedLengths[i]))); \
    } \
    else \
    { \
        ret.m_Idx[i] = 0; \
    }

    ___addidx(0);
    ___addidx(1);
    ___addidx(2);
    ___addidx(3);
    ___addidx(4);
    ___addidx(5);
    ___addidx(6);
    ___addidx(7);

    #undef ___addidx

    return ret;
}

/**
* map working indice to linear tensor index
*/
__device__ static __inline__ UINT _deviceWorkIndexToTensorIndexNaive(
    const smallIdx* __restrict__ wkidx,
    const UINT* __restrict__ strides,
    const UINT start,
    BYTE byOrder)
{
    UINT uiRet = start;
    //#pragma unroll
    //for (BYTE i = 0; i < _kSmallOrder; ++i)
    //{
    //    if (i < byOrder)
    //    {
    //        uiRet = uiRet + (wkidx[i] + starts[i]) * strides[i];
    //    }
    //}
    //unroll in .h file is unknown pragma

    #define ___addidx(i) \
    if (i < byOrder) \
    { \
        uiRet = uiRet + wkidx[i] * strides[i]; \
    }

    ___addidx(0);
    ___addidx(1);
    ___addidx(2);
    ___addidx(3);
    ___addidx(4);
    ___addidx(5);
    ___addidx(6);
    ___addidx(7);

    #undef ___addidx

    return uiRet;
}

#pragma endregion

#if 0
class CNAPI CNDeviceTensorCommonNaive : public TCNDeviceTensorCommon<CNDeviceTensorCommonNaive>
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
        BYTE byIndexCount);

    template<class srcT, class valT>
    void FillMasked(CNDeviceTensor<srcT>* src, const valT& val,
        const UINT* __restrict__ srcIndexStart,
        const BYTE* __restrict__ mask,
        const UINT* __restrict__ maskStride,
        const UINT* __restrict__ maskIndexStart,
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

#endif//#ifndef _CNDEVICETENSOR_COMMON_NAIVE_H_

//=============================================================================
// END OF FILE
//=============================================================================