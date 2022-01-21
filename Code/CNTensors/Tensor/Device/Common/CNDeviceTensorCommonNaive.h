//=============================================================================
// FILENAME : CNDeviceTensorCommonNaive.h
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
    hostBuffer[byIndexCount - 1] = 1; \
    for (INT i = byIndexCount - 2; i >= 0; --i) /* do not use BYTE here*/ \
    { \
        hostBuffer[i] = hostBuffer[i + 1] * lengths[i + 1]; \
    } \
    _memcpy_hd(ptr, hostBuffer, dataSize);

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

//__CN_FORCEOBJ_HEAD(CNDeviceTensorCommonNaive);

template<class Operator, class T>
class __DLL_EXPORT CNDeviceTensorCommonNaiveOneOperator
: public TCNDeviceTensorCommonOneOperator<CNDeviceTensorCommonNaiveOneOperator<Operator, T>, Operator, T>
{
public:

    CNDeviceTensorCommonNaiveOneOperator(T* pBuffer)
        : TCNDeviceTensorCommonOneOperator<CNDeviceTensorCommonNaiveOneOperator, Operator, T>(pBuffer)
    {
        
    }

    /**
     * This dummy function is only to put the implementations into .obj files
     */
    //void __Dummy()
    //{
    //    OneOperator(TOperator_Zero<T>(), 0, NULL, NULL, 0, 0);
    //    OneOperator(TOperator_One<T>(), 0, NULL, NULL, 0, 0);
    //}

    //void Set(
    //    CNDeviceTensor<T>* dst, const T& v,
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    appCrucial(_T("Not implemented yet...\n"));
    //}

    //void Zero(
    //    CNDeviceTensor<T>* dst,
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    TOperator_Zero<T> op;
    //    OneOperator(op, dst, dstIndexStart, dstStride, lengths, byIndexCount);
    //}

    //void One(
    //    CNDeviceTensor<T>* dst,
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    TOperator_One<T> op;
    //    OneOperator(op, dst, dstIndexStart, dstStride, lengths, byIndexCount);
    //}



    //friend class TCNDeviceTensorCommon<CNDeviceTensorCommonNaive, T>;

//public:

    void OneOperator(
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount) override;



    //template<class Operator>
    //void OneOperatorD(
    //    TOperator_D<Operator, T> op,
    //    CNDeviceTensor<T>* dst,
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    CNDeviceTensor<T>* src,
    //    const UINT srcIndexStart,
    //    const UINT* __restrict__ srcStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    appCrucial(_T("Not implemented yet...\n"));
    //}

};

//extern CNDeviceTensorCommonNaive<INT> GCalculatorNaiveCommon;
//extern CNDeviceTensorCommonNaive<FLOAT> GCalculatorNaiveCommon;
//extern CNDeviceTensorCommonNaive<DOUBLE> GCalculatorNaiveCommonDouble;
//extern CNDeviceTensorCommonNaive<INT> GCalculatorNaiveCommon;

//template<class T, class Operator>
//void Calc_OneOperator(
//    const CNDeviceTensor<T>& tensor,
//    const CNDeviceTensorCommonNaive& calc,
//    const UINT dstIndexStart,
//    const UINT* __restrict__ dstStride,
//    const UINT* __restrict__ lengths,
//    BYTE byIndexCount)
//{
//    Operator opone;
//    calc.OneOperator(opone, tensor, dstIndexStart, dstStride, lengths, byIndexCount);
//}

//template class CNDeviceTensorCommonNaive<_SComplex>;

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_NAIVE_H_

//=============================================================================
// END OF FILE
//=============================================================================