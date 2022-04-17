//=============================================================================
// FILENAME : CNDeviceTensor.h
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [18/06/2021 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_H_
#define _CNDEVICETENSOR_H_

#define __DeviceTensorOneElementFunc(name) \
template <class Calc> \
void name( \
    TCNDeviceTensorCommon<Calc>* pCalc, \
    const UINT dstIndexStart, \
    const UINT* __restrict__ dstStride, \
    const UINT* __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    pCalc->name( \
        m_pDeviceDataBuffer, \
        dstIndexStart, \
        dstStride, \
        lengths, \
        byIndexCount); \
}

#define __DeviceTensorTwoElementFunc(name) \
template <class Calc, class Tsrc> \
void name( \
    TCNDeviceTensorCommon<Calc>* pCalc, \
    const Tsrc& v, \
    const UINT dstIndexStart, \
    const UINT* __restrict__ dstStride, \
    const UINT* __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    pCalc->name( \
        m_pDeviceDataBuffer, \
        v, \
        dstIndexStart, \
        dstStride, \
        lengths, \
        byIndexCount); \
}


__BEGIN_NAMESPACE

__DEFINE_ENUM(ECalculator,
    EC_Naive,
    )

/**
 * A 32 order tensor with all dim=2 needs 64G memory
 * Any tensor larger than this is not capable
 */
//constexpr BYTE _kMaxSupportedOrder = 32;

class CNAPI CNDeviceTensorPlaceHolder
{
public:
    virtual ~CNDeviceTensorPlaceHolder() {}
};

/**
* It is designed to work like this:
*
* device_tensor1 = host_tensor1.GetDeviceTensor()
* device_tensor2 = host_tensor2.GetDeviceTensor()
* device_tensor3 = contractor.contract(device_tensor1, device_tensor2, other_parameters)
* host_tensor3 = create_host_tensor(device_tensor3)
*
*/
template<class T>
class __DLL_EXPORT CNDeviceTensor //: public CNDeviceTensorPlaceHolder
{
public:

    CNDeviceTensor()
        : m_pDeviceDataBuffer(NULL)
        , m_pDeviceStrides(NULL)
        , m_pDeviceLength(NULL)
        , m_iDim(0)
    {
        
    }

    ~CNDeviceTensor()
    {
        Release();
    }

    void Release()
    {
        appCudaFree(m_pDeviceDataBuffer);
        appCudaFree(m_pDeviceStrides);
        appCudaFree(m_pDeviceLength);
    }

    void CreateEmpty(const UINT* lengths, UINT dim)
    {
        Release();
        m_iDim = dim;
        UINT* strides = (UINT*)appAlloca(sizeof(UINT) * dim);
        strides[dim - 1] = 1;
        UINT v = 1;
        for (UINT i = 1; i < dim; ++i)
        {
            v = v * lengths[dim - i];
            strides[dim - i - 1] = v;
        }
        v = v * lengths[0];

        appCudaMalloc((void**)&m_pDeviceDataBuffer, sizeof(T) * v);
        appCudaMalloc((void**)&m_pDeviceStrides, sizeof(UINT) * dim);
        appCudaMalloc((void**)&m_pDeviceLength, sizeof(UINT) * dim);

        _memcpy_hd(m_pDeviceStrides, strides, sizeof(UINT) * dim);
        _memcpy_hd(m_pDeviceLength, lengths, sizeof(UINT) * dim);
    }

    void DebugPrint(UINT uiXDim, UINT uiYDim) const
    {
        CNDeviceTensorCommonEmpty::DebugPrint(m_pDeviceDataBuffer, uiXDim, uiYDim);
    }

    template <class Calc, class Tsrc>
    void Set(
        TCNDeviceTensorCommon<Calc>* pCalc,
        const Tsrc& v,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        pCalc->Set(
            m_pDeviceDataBuffer,
            v,
            dstIndexStart,
            dstStride,
            lengths,
            byIndexCount);
    }

    template <class Calc, class Tsrc>
    void Set(
        TCNDeviceTensorCommon<Calc>* pCalc,
        const Tsrc* __restrict__ src,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        pCalc->Set(
            m_pDeviceDataBuffer,
            src,
            dstIndexStart,
            dstStride,
            srcIndexStart,
            srcStride,
            lengths,
            byIndexCount);
    }

    __OVER_ALL_ONE_OP(__DeviceTensorOneElementFunc)

    __OVER_ALL_TWO_OP(__DeviceTensorTwoElementFunc)

    template <class Calc, class Tsrc>
    void Add(
        TCNDeviceTensorCommon<Calc>* pCalc,
        const Tsrc* __restrict__ v,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        pCalc->Add(
            m_pDeviceDataBuffer,
            v,
            dstIndexStart,
            dstStride,
            srcIndexStart,
            srcStride,
            lengths,
            byIndexCount);
    }

    T* m_pDeviceDataBuffer;
    UINT* m_pDeviceStrides;
    UINT* m_pDeviceLength;
    UINT m_iDim;

};


__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_H_

//=============================================================================
// END OF FILE
//=============================================================================