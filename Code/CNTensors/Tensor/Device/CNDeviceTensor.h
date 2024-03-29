//=============================================================================
// FILENAME : CNDeviceTensor.h
// 
// DESCRIPTION:
// 
//
// REVISION[d-m-y]:
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


#define __DeviceTensorOneElementFuncTotal(name) \
template <class Calc> \
void name(TCNDeviceTensorCommon<Calc>* pCalc) \
{ \
    pCalc->name(m_pDeviceDataBuffer, m_uiTotalSize); \
}


#define __DeviceTensorTwoElementFuncValue(name) \
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


#define __DeviceTensorTwoElementFuncTensor(name) \
template <class Calc, class Tsrc> \
void name( \
    TCNDeviceTensorCommon<Calc>* pCalc, \
    const Tsrc* __restrict__ v, \
    const UINT dstIndexStart, \
    const UINT* __restrict__ dstStride, \
    const UINT srcIndexStart, \
    const UINT* __restrict__ srcStride, \
    const UINT* __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    pCalc->name( \
        m_pDeviceDataBuffer, \
        v, \
        dstIndexStart, \
        dstStride, \
        srcIndexStart, \
        srcStride, \
        lengths, \
        byIndexCount); \
}


#define __DeviceTensorTwoElementFuncTotal(name) \
template <class Calc, class Tsrc> \
void name(TCNDeviceTensorCommon<Calc>* pCalc, const Tsrc& v) \
{ \
    pCalc->name(m_pDeviceDataBuffer, v, m_uiTotalSize); \
}


__BEGIN_NAMESPACE

template<class T> __inline__ __host__ __device__
BYTE TensorRtti(T* pointer) { return 0; }

template<> __inline__ __host__ __device__
BYTE TensorRtti(SBYTE* pointer) { return 1; }

template<> __inline__ __host__ __device__
BYTE TensorRtti(INT* pointer) { return 2; }

template<> __inline__ __host__ __device__
BYTE TensorRtti(FLOAT* pointer) { return 3; }

template<> __inline__ __host__ __device__
BYTE TensorRtti(DOUBLE* pointer) { return 4; }

template<> __inline__ __host__ __device__
BYTE TensorRtti(_SComplex* pointer) { return 5; }

template<> __inline__ __host__ __device__
BYTE TensorRtti(_DComplex* pointer) { return 6; }


UINT __inline__ __host__ __device__ 
GetTypeSizeOf(BYTE type)
{
    switch (type)
    {
    case 1:
        return static_cast<UINT>(sizeof(SBYTE));
    case 2:
        return static_cast<UINT>(sizeof(INT));
    case 3:
        return static_cast<UINT>(sizeof(FLOAT));
    case 4:
        return static_cast<UINT>(sizeof(DOUBLE));
    case 5:
        return static_cast<UINT>(sizeof(FLOAT) * 2);
    case 6:
        return static_cast<UINT>(sizeof(DOUBLE) * 2);
    }
    return 0;
}

template<class T> __inline__ __host__ __device__
UBOOL TensorTypeConvert(T& dst, BYTE type, BYTE* valueAddress) 
{ 
    switch (type)
    {
    case 1:
        {
            SBYTE v = 0;
            memcpy(&v, valueAddress, sizeof(SBYTE));
            TOperator_Set<T, SBYTE> setop;
            setop.Do(dst, v);
            return TRUE;
        }
        break;
    case 2:
        {
            INT v = 0;
            memcpy(&v, valueAddress, sizeof(INT));
            TOperator_Set<T, INT> setop;
            setop.Do(dst, v);
            return TRUE;
        }
        break;
    case 3:
        {
            FLOAT v = 0.0f;
            memcpy(&v, valueAddress, sizeof(FLOAT));
            TOperator_Set<T, FLOAT> setop;
            setop.Do(dst, v);
            return TRUE;
        }
        break;
    case 4:
        {
            DOUBLE v = 0.0;
            memcpy(&v, valueAddress, sizeof(DOUBLE));
            TOperator_Set<T, DOUBLE> setop;
            setop.Do(dst, v);
            return TRUE;
        }
        break;
    case 5:
        {
            _SComplex v = _zerocs;
            //memcpy(&v.x, valueAddress, sizeof(FLOAT));
            //memcpy(&v.y, valueAddress + sizeof(FLOAT), sizeof(FLOAT));
            memcpy(&v, valueAddress, sizeof(_SComplex));
            TOperator_Set<T, _SComplex> setop;
            setop.Do(dst, v);
            return TRUE;
        }
        break;
    case 6:
        {
            _DComplex v = _zerocd;
            //memcpy(&v.x, valueAddress, sizeof(DOUBLE));
            //memcpy(&v.y, valueAddress + sizeof(DOUBLE), sizeof(DOUBLE));
            memcpy(&v, valueAddress, sizeof(_DComplex));
            TOperator_Set<T, _DComplex> setop;
            setop.Do(dst, v);
            return TRUE;
        }
        break;
    }
    return FALSE; 
}


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
class __DLL_EXPORT CNDeviceTensor
{
public:

    CNDeviceTensor()
        : m_pDeviceDataBuffer(NULL)
        , m_uiTotalSize(0)
    {
        
    }

    ~CNDeviceTensor()
    {
        Release();
    }

    void Release()
    {
        m_uiTotalSize = 0;
        appCudaFree(m_pDeviceDataBuffer);
    }

    UBOOL CreateEmpty(UINT uiVolumn)
    {
        if (uiVolumn != m_uiTotalSize)
        {
            Release();
            m_uiTotalSize = uiVolumn;
            appCudaMalloc((void**)&m_pDeviceDataBuffer, sizeof(T) * uiVolumn);
            return TRUE;
        }
        return FALSE;
    }

    void DebugPrint(UINT uiXDim, UINT uiYDim) const
    {
        CNDeviceTensorCommonEmpty::DebugPrint(m_pDeviceDataBuffer, m_uiTotalSize, uiXDim, uiYDim);
    }

    void CopyOut(BYTE* buffer) const
    {
        _memcpy_dh(buffer, m_pDeviceDataBuffer, sizeof(T) * m_uiTotalSize);
    }

    void CopyIn(BYTE* buffer)
    {
        _memcpy_hd(m_pDeviceDataBuffer, buffer, sizeof(T) * m_uiTotalSize);
    }

    template <class Calc, class Tsrc>
    void Set(TCNDeviceTensorCommon<Calc>* pCalc, const Tsrc& v)
    {
        pCalc->Set(m_pDeviceDataBuffer, v, m_uiTotalSize);
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

    template <class Calc>
    void Random(TCNDeviceTensorCommon<Calc>* pCalc, UINT uiRandomType)
    {
        pCalc->Random(m_pDeviceDataBuffer, uiRandomType, m_uiTotalSize);
    }

    template <class Calc>
    void Random(TCNDeviceTensorCommon<Calc>* pCalc, 
        UINT uiRandomType,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        pCalc->Random(m_pDeviceDataBuffer, uiRandomType, dstIndexStart, dstStride, lengths, byIndexCount);
    }

    __OVER_ALL_ONE_OP(__DeviceTensorOneElementFunc)

    __OVER_ALL_ONE_OP(__DeviceTensorOneElementFuncTotal)

    __OVER_ALL_TWO_OP(__DeviceTensorTwoElementFuncValue)

    __OVER_ALL_TWO_OP(__DeviceTensorTwoElementFuncTensor)

    __OVER_ALL_TWO_OP(__DeviceTensorTwoElementFuncTotal)
        
    template <class Calc, class Tsrc>
    void Axpy(
        TCNDeviceTensorCommon<Calc>* pCalc,
        const T& v,
        const Tsrc* __restrict__ src,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        pCalc->Axpy(
            m_pDeviceDataBuffer,
            v,
            src,
            dstIndexStart,
            dstStride,
            srcIndexStart,
            srcStride,
            lengths,
            byIndexCount);
    }

    template <class Calc>
    void Sum(TCNDeviceTensorContraction<Calc>* pCalc,
        const T* __restrict__ pSrcBuffer,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount,
        UINT uiSumLength,
        UINT uiSumIndexStride)
    {
        pCalc->Sum(m_pDeviceDataBuffer, pSrcBuffer, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount, uiSumLength, uiSumIndexStride);
    }

    template <class Calc>
    void Prod(TCNDeviceTensorContraction<Calc>* pCalc,
        const T* __restrict__ pSrcBuffer,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount,
        UINT uiSumLength,
        UINT uiSumIndexStride)
    {
        pCalc->Prod(m_pDeviceDataBuffer, pSrcBuffer, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount, uiSumLength, uiSumIndexStride);
    }

    /**
    * Note that reduce will mass up the tensor
    */
    template <class Calc>
    T ReduceSum(TCNDeviceTensorContraction<Calc>* pCalc)
    {
        return pCalc->ReduceSum(m_pDeviceDataBuffer, m_uiTotalSize);
    }

    template <class Calc>
    T ReduceProd(TCNDeviceTensorContraction<Calc>* pCalc)
    {
        return pCalc->ReduceProd(m_pDeviceDataBuffer, m_uiTotalSize);
    }

    template <class Calc, class srcT>
    void Contraction(TCNDeviceTensorContraction<Calc>* pCalc, const T* src1, const srcT* src2,
        UINT uiDstIndexStart, UINT* dstStrides, UINT uiSrc1IndexStart, UINT uiSrc2IndexStart, UINT* src1Strides, UINT* src2Strides,
        UINT* lengths, BYTE byIdxCount, BYTE byLeftIdxCount,
        UINT uiSumLength, UINT uiSumStrideLeft, UINT uiSumStrideRight)
    {
        pCalc->Contraction(m_pDeviceDataBuffer, src1, src2, uiDstIndexStart, dstStrides,
            uiSrc1IndexStart, uiSrc2IndexStart, src1Strides, src2Strides, 
            lengths, byIdxCount, byLeftIdxCount, uiSumLength, uiSumStrideLeft, uiSumStrideRight);
    }

    template <class Calc, class srcT>
    void Contraction(TCNDeviceTensorContraction<Calc>* pCalc, const T* src1, const srcT* src2,
        UINT uiDstIndexStart, UINT* dstStrides, UINT uiSrc1IndexStart, UINT uiSrc2IndexStart, UINT* src1Strides, UINT* src2Strides,
        UINT* lengths, BYTE byIdxCount, BYTE byLeftIdxCount,
        UINT* uiSumStrideLeft, UINT* uiSumStrideRight, UINT* uiSumLength, BYTE bySumIndexCount)
    {
        pCalc->Contraction(m_pDeviceDataBuffer, src1, src2, uiDstIndexStart, dstStrides,
            uiSrc1IndexStart, uiSrc2IndexStart, src1Strides, src2Strides,
            lengths, byIdxCount, byLeftIdxCount, 
            uiSumStrideLeft, uiSumStrideRight, uiSumLength, bySumIndexCount);
    }

    T* m_pDeviceDataBuffer;
    UINT m_uiTotalSize;

};


__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_H_

//=============================================================================
// END OF FILE
//=============================================================================