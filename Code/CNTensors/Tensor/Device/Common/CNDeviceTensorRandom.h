//=============================================================================
// FILENAME : CNDeviceTensorRandom.h
// 
// DESCRIPTION:
// 
// Meanings of three indexes
//
// This is not likely to be inheriented.
// It generate random numbers, called by CNDeviceTensorCommonNaive
//
// REVISION:
//  [19/04/2022 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_RANDOM_H_
#define _CNDEVICETENSOR_RANDOM_H_

#define __IMPLEMENT_COMMON_RANDOM(type) \
template class TCNDeviceTensorRandom<type>;

__BEGIN_NAMESPACE

template<class T>
class __DLL_EXPORT TCNDeviceTensorRandom
{
public:

    void Random(T* pBuffer, UINT uiRandomType, UINT uiTotalSize);

    void Random(
        T* pBuffer,
        UINT uiRandomType,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount);
};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_NAIVE_H_

//=============================================================================
// END OF FILE
//=============================================================================