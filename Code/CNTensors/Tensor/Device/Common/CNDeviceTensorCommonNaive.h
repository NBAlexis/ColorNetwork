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


#define __NAIVECALC_ONEELEMENT(name) \
template<class T> \
void name( \
    T * pBuffer, \
    UINT dstIndexStart, \
    const UINT * __restrict__ dstStride, \
    const UINT * __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    CNDeviceTensorCommonNaiveOneOperator<TOperator_##name<T>, T>() \
        .OneOperator(pBuffer, dstIndexStart, dstStride, lengths, byIndexCount); \
}


#define __NAIVECALC_TWOELEMENT_VALUE(name) \
template<class Tdst, class Tsrc> \
void name( \
    Tdst * pBuffer, \
    const Tsrc & v, \
    UINT dstIndexStart, \
    const UINT * __restrict__ dstStride, \
    const UINT * __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    CNDeviceTensorCommonTwoOperatorNaive<TOperator_##name<Tdst, Tsrc>, Tdst, Tsrc>() \
        .TwoOperatorValue(pBuffer, v, dstIndexStart, dstStride, lengths, byIndexCount); \
}


#define __NAIVECALC_TWOELEMENT_TENSOR(name) \
template<class Tdst, class Tsrc> \
void name( \
    Tdst * pBuffer, \
    const Tsrc * __restrict__ v, \
    UINT dstIndexStart, \
    const UINT * __restrict__ dstStride, \
    UINT srcIndexStart, \
    const UINT * __restrict__ srcStride, \
    const UINT * __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    CNDeviceTensorCommonTwoOperatorNaive<TOperator_##name<Tdst, Tsrc>, Tdst, Tsrc>() \
        .TwoOperatorTensor(pBuffer, v, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount); \
}

#define __NAIVECALC_ONEELEMENT_TOTAL(name) \
template<class Tdst> \
void name(Tdst * pBuffer, UINT uiTotalSize) \
{ \
    CNDeviceTensorCommonNaiveOneOperator<TOperator_##name<Tdst>, Tdst>().OneOperator(pBuffer, uiTotalSize); \
}


#define __NAIVECALC_TWOELEMENT_TOTAL(name) \
template<class Tdst, class Tsrc> \
void name(Tdst * pBuffer, const Tsrc & v, UINT uiTotalSize) \
{ \
    CNDeviceTensorCommonTwoOperatorNaive<TOperator_##name<Tdst, Tsrc>, Tdst, Tsrc>().TwoOperatorValueTotal(pBuffer, v, uiTotalSize); \
}

__BEGIN_NAMESPACE

template<class Operator, class T>
class __DLL_EXPORT CNDeviceTensorCommonNaiveOneOperator
{
public:

    CNDeviceTensorCommonNaiveOneOperator()
        : m_op()
    {
        
    }

    void OneOperator(
        T* pBuffer,
        UINT totalSize);

    void OneOperator(
        T* pBuffer,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount);

protected:

    TOperator_D<Operator, T> m_op;
};

template<class Operator, class Tdst, class Tsrc>
class __DLL_EXPORT CNDeviceTensorCommonTwoOperatorNaive
{
public:
    CNDeviceTensorCommonTwoOperatorNaive()
        : m_op()
    {

    }

    void TwoOperatorValueTotal(
        Tdst* pBuffer,
        const Tsrc& v,
        UINT uiTotalSize);

    void TwoOperatorValue(
        Tdst* pBuffer,
        const Tsrc& v,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount);

    void TwoOperatorTensor(
        Tdst* pBuffer,
        const Tsrc* __restrict__ src,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount);

protected:

    TOperator_S<Operator, Tdst, Tsrc> m_op;
};

class __DLL_EXPORT CNDeviceTensorCommonNaive : public TCNDeviceTensorCommon<CNDeviceTensorCommonNaive>
{
public:

    template<class Tdst, class Tsrc>
    void Set(Tdst* pBuffer, const Tsrc& v, UINT uiTotalSize)
    {
        CNDeviceTensorCommonTwoOperatorNaive<TOperator_Set<Tdst, Tsrc>, Tdst, Tsrc>().TwoOperatorValueTotal(pBuffer, v, uiTotalSize);
    }

    template<class Tdst, class Tsrc>
    void Set(
        Tdst* pBuffer,
        const Tsrc& v,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        CNDeviceTensorCommonTwoOperatorNaive<TOperator_Set<Tdst, Tsrc>, Tdst, Tsrc>()
            .TwoOperatorValue(pBuffer, v, dstIndexStart, dstStride, lengths, byIndexCount);
    }

    template<class Tdst, class Tsrc>
    void Set(
        Tdst* pBuffer,
        const Tsrc* __restrict__ src,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        CNDeviceTensorCommonTwoOperatorNaive<TOperator_Set<Tdst, Tsrc>, Tdst, Tsrc>()
            .TwoOperatorTensor(pBuffer, src, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount);
    }

    template<class Tdst>
    void Random(Tdst* pBuffer, UINT uiRandomType, UINT uiTotalSize)
    {
        TCNDeviceTensorRandom<Tdst>().Random(pBuffer, uiRandomType, uiTotalSize);
    }

    template<class Tdst>
    void Random(
        Tdst* pBuffer,
        UINT uiRandomType,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        TCNDeviceTensorRandom<Tdst>().One(pBuffer, uiRandomType, dstIndexStart, dstStride, lengths, byIndexCount);
    }

    __OVER_ALL_ONE_OP(__NAIVECALC_ONEELEMENT)

    __OVER_ALL_ONE_OP(__NAIVECALC_ONEELEMENT_TOTAL)
        
    __OVER_ALL_TWO_OP(__NAIVECALC_TWOELEMENT_VALUE)

    __OVER_ALL_TWO_OP(__NAIVECALC_TWOELEMENT_TENSOR)

    __OVER_ALL_TWO_OP(__NAIVECALC_TWOELEMENT_TOTAL)
        
};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_NAIVE_H_

//=============================================================================
// END OF FILE
//=============================================================================