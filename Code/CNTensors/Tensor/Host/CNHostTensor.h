//=============================================================================
// FILENAME : CNHostTensor.h
// 
// DESCRIPTION:
// Host tensor should be a template class without template function
//
// REVISION:
//  [10/31/2021 nbalexis]
//=============================================================================
#ifndef _CNHOSTTENSOR_H_
#define _CNHOSTTENSOR_H_

#define __TEST_HOST_DEVECE_ONE_ELEMENT_INTERFACE(name) \
template<class Calc> \
void name(TCNDeviceTensorCommon<Calc>& calc, UINT uiIndexStart, const UINT* strides, const UINT* lengths, BYTE uiIndexCount) \
{ \
    m_cDeviceTensor.name(&calc, uiIndexStart, strides, lengths, uiIndexCount); \
}

#define __TEST_HOST_DEVECE_TWO_ELEMENT_INTERFACE(name) \
template<class Calc, class Tsrc> \
void name(TCNDeviceTensorCommon<Calc>& calc, const Tsrc& v, UINT uiIndexStart, const UINT* strides, const UINT* lengths, BYTE uiIndexCount) \
{ \
    m_cDeviceTensor.name(&calc, v, uiIndexStart, strides, lengths, uiIndexCount); \
}

__BEGIN_NAMESPACE

template<class T>
class __DLL_EXPORT CNHostTensor //: public CNDeviceTensorPlaceHolder
{
public:

    CNHostTensor()
        : m_pStrides(NULL)
        , m_pLength(NULL)
        , m_iDim(0)
    {

    }

    ~CNHostTensor()
    {
        Release();
    }

    void Release()
    {
        m_cDeviceTensor.Release();
        appSafeFree(m_pStrides);
        appSafeFree(m_pLength);
    }

    void CreateEmpty(const UINT* lengths, UINT dim)
    {
        Release();

        m_pStrides = (UINT*)malloc(sizeof(UINT) * dim);
        m_pStrides[dim - 1] = 1;
        UINT v = 1;
        for (UINT i = 1; i < dim; ++i)
        {
            v = v * lengths[dim - i];
            m_pStrides[dim - i - 1] = v;
        }
        m_iDim = dim;
        m_pLength = (UINT*)malloc(sizeof(UINT) * dim);
        memcpy(m_pLength, lengths, sizeof(UINT) * dim);

        m_cDeviceTensor.CreateEmpty(m_pLength, m_iDim);
    }

    void DebugPrint(UINT uiXDim, UINT uiYDim) const
    {
        m_cDeviceTensor.DebugPrint(uiXDim, uiYDim);
    }

    template<class Calc, class Tsrc>
    void Set(TCNDeviceTensorCommon<Calc>& calc, const Tsrc& v, UINT uiIndexStart, const UINT* strides, const UINT* lengths, BYTE uiIndexCount)
    {
        m_cDeviceTensor.Set(&calc, v, uiIndexStart, strides, lengths, uiIndexCount);
    }

    template<class Calc, class Tsrc>
    void Set(TCNDeviceTensorCommon<Calc>& calc, const CNHostTensor<Tsrc>& v,
        UINT uiIndexStart,
        const UINT* strides,
        UINT uiSrcIndexStart,
        const UINT* srcStrides,
        const UINT* lengths,
        BYTE uiIndexCount)
    {
        m_cDeviceTensor.Set(&calc, v.m_cDeviceTensor.m_pDeviceDataBuffer,
            uiIndexStart, strides,
            uiSrcIndexStart, srcStrides,
            lengths, uiIndexCount);
    }

    __OVER_ALL_ONE_OP(__TEST_HOST_DEVECE_ONE_ELEMENT_INTERFACE)

    __OVER_ALL_TWO_OP(__TEST_HOST_DEVECE_TWO_ELEMENT_INTERFACE)

    template<class Calc, class Tsrc>
    void Add(TCNDeviceTensorCommon<Calc>& calc, const CNHostTensor<Tsrc>& v,
        UINT uiIndexStart, 
        const UINT* strides, 
        UINT uiSrcIndexStart,
        const UINT* srcStrides,
        const UINT* lengths, 
        BYTE uiIndexCount)
    {
        m_cDeviceTensor.Add(&calc, v.m_cDeviceTensor.m_pDeviceDataBuffer, 
            uiIndexStart, strides, 
            uiSrcIndexStart, srcStrides,
            lengths, uiIndexCount);
    }

    CNDeviceTensor<T> m_cDeviceTensor;
    UINT* m_pStrides;
    UINT* m_pLength;
    UINT m_iDim;

protected:

    //template<class Operator>
    //void OneOperator(
    //    ECalculator eCalc,
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    m_cDeviceTensor.template OneOperator<Operator>(
    //        eCalc,
    //        dstIndexStart,
    //        dstStride,
    //        lengths,
    //        byIndexCount
    //        );
    //}
};


__END_NAMESPACE

#endif//#ifndef _CNHOSTTENSOR_H_

//=============================================================================
// END OF FILE
//=============================================================================