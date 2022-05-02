//=============================================================================
// FILENAME : CNDeviceTensorNaiveIndexMapping.h
// 
// DESCRIPTION:
// 
// Meanings of three indexes
//
// Thread Index is the thread idx different for each thread and start from 0
// Tensor Index is (linear) index of the buffer, pBuffer[tensorIndex]
// WorkIndex is human readable tensor index, like {1,2,3,4,0,0,0,0} for T[1,2,3,4]
//
// REVISION:
//  [19/04/2022 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_NAIVE_INDEXMAPPING_H_
#define _CNDEVICETENSOR_NAIVE_INDEXMAPPING_H_

__BEGIN_NAMESPACE


/**
 * In the case of TN, typically we have a tensor smaller than this
 */
constexpr BYTE _kSmallOrder = 8;
constexpr USHORT _kSmallMaxDim = 256;
typedef USHORT smallIdx;

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

inline TArray<UINT> GetFactors(UINT length)
{
    TArray<UINT> ret;
    ret.AddItem(1);
    for (UINT i = 2; i < (length / 2); ++i)
    {
        if ((i * (length / i)) == length)
        {
            ret.AddItem(i);
        }
    }
    if (length > 1)
    {
        ret.AddItem(length);
    }
    return ret;
}

inline void GetDecompose(UINT uiConstraint, UINT toDecompose, UINT& block, UINT& thread)
{
    block = toDecompose;
    thread = 1;

    TArray<UINT> factors = GetFactors(toDecompose);
    for (INT i = 0; i < factors.Num(); ++i)
    {
        if (factors[i] <= uiConstraint)
        {
            thread = factors[i];
            block = toDecompose / factors[i];
        }
        else
        {
            return;
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
         * when i = 3,      4 = 1234 % 10
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
* This is for two index
* outL = left start index
* outR = right start index
* 
* For thread index 1234, we need
*
* when L=2
* outL = outL + 1 * stride[0] + 2 * stride[1]
* outR = outR + 3 * stride[2] + 3 * stride[4]
*
* or
* when L=1
* outL = outL + 1 * stride[0]
* outR = outR + 2 * stride[1] + 3 * stride[2] + 3 * stride[4]
*/
__device__ static __inline__ void _deviceThreadIdxToTensorIdxNaiveLR(
    UINT& outL,
    UINT& outR,
    BYTE byL,
    const UINT* __restrict__ strides,
    const UINT* __restrict__ multpliedLengths,
    UINT uiIdx, BYTE byOrder)
{

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

        if (i < byL)
        {
            outL += idxForThisOrder * strides[i];
        }
        else
        {
            outR += idxForThisOrder * strides[i];
        }
    }
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

/**
* map working indice to linear tensor index
*/
__device__ static __inline__ void _deviceWorkIndexToTensorIndexNaiveLR(
    UINT& outL,
    UINT& outR,
    BYTE byLeft,
    const smallIdx* __restrict__ wkidx,
    const UINT* __restrict__ strides,
    BYTE byOrder)
{
    //#pragma unroll
    //for (BYTE i = 0; i < _kSmallOrder; ++i)
    //{
    //    if (i < byLeft)
    //    {
    //        outL = outL + wkidx[i] * strides[i];
    //    }
    //    else if (i < byOrder)
    //    {
    //        outR = outR + wkidx[i] * strides[i];
    //    }
    //}
    //unroll in .h file is unknown pragma

#define ___addidx(i) \
    if (i < byLeft) \
    { \
        outL = outL + wkidx[i] * strides[i]; \
    } \
    else if (i < byOrder) \
    { \
        outR = outR + wkidx[i] * strides[i]; \
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

}


__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_NAIVE_H_

//=============================================================================
// END OF FILE
//=============================================================================