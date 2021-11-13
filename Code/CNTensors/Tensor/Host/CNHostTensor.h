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

    //static CNCalcTensorCommon* GetCommonCalculator(ECalculator eCalc)
    //{
    //    switch (eCalc)
    //    {
    //    case EC_Naive:
    //        return &GCalculatorNaiveCommon;
    //    }

    //    const CCString sCalculatorName = __ENUM_TO_STRING(ECalculator, eCalc);
    //    appCrucial(_T("Common Calculator not implementd! %s\n"), sCalculatorName.c_str());
    //    return NULL;
    //}

    void DebugPrint(UINT uiXDim, UINT uiYDim) const
    {
        TCNDeviceTensorCommon<CNDeviceTensorCommonNaive<T>, T>::DebugPrint(m_cDeviceTensor, uiXDim, uiYDim);
    }

    void Zero(ECalculator eCalc, UINT uiIndexStart, const UINT* strides, const UINT* lengths, BYTE uiIndexCount);
    void One(ECalculator eCalc, UINT uiIndexStart, const UINT* strides, const UINT* lengths, BYTE uiIndexCount);

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