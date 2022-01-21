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


/**
 * This is called CRTP, see:
 * https://stackoverflow.com/questions/2354210/can-a-class-member-function-template-be-virtual#
 *
 * To implement a calculator, call
 * class Calculator : public TCNDeviceTensorCommon<Calculator>
 *
 * NOTE!!!:
 *  this template is just for convience to make sure all types using one code
 *  pay attention here:
 *    https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
 *  The only portable way of using templates at the moment is to implement them in header files by using inline functions.
 *  We will GIVE UP "portable" when we met this problem
 *
 * Meanings of "index start", "strides" and "lengths"
 *
 * Let, m:n = {m, m+1, ..., n-1}
 * We need to do things like:
 *   T1[..., a0:(a0+length0), ..., a1:(a1+length1), ..., ai:(ai+lengthi), ...]
 *     = T2[..., b0:(b0+length0), ..., b1:(b1+length1), ..., bi:(bi+lengthi), ...]
 *
 * "byIndexCount" = i
 * the size of "dstStride" and "lengths" is also i
 * "dstStride" controlls the position of index,
 *   for example T[0, 2:5], stride=1,
 *   T[2:5, 0], stride is size of the first index
 *
 * dstIndexStart = sum_i ai*stridei, controlls a0,a1,...
 */
template<class T>
class __DLL_EXPORT TCNDeviceTensorCommon
{
public:

    //TCNDeviceTensorCommon(T* pBuffer)
    //    : m_pBuffer(pBuffer)
    //{
    //    
    //}

    //virtual ~TCNDeviceTensorCommon()
    //{
    //    
    //}

    /**
     * Things like a = constant
     */
    //void Set(
    //    CNDeviceTensor<T>* dst, const T& v,
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    ((Calculator*)this)->Set(dst, v, dstIndexStart, dstStride, lengths, byIndexCount);
    //}

    //void Zero(
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    TOperator_Zero<T> op;
    //    OneOperator(op, m_pBuffer, dstIndexStart, dstStride, lengths, byIndexCount);
    //}

    //void One(
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    TOperator_One<T> op;
    //    OneOperator(op, m_pBuffer, dstIndexStart, dstStride, lengths, byIndexCount);
    //}

    static void DebugPrint(const CNDeviceTensor<T>& src, UINT uiSize)
    {
        appPushLogDate(FALSE);
        T* hostBuffer = (T*)malloc(sizeof(T) * uiSize);
        appGetCuda()->CopyDH(hostBuffer, src.m_pDeviceDataBuffer, sizeof(T) * uiSize);
        for (UINT i = 0; i < uiSize; ++i)
        {
            appGeneral(_T("%d: "), i);
            LogValue(hostBuffer[i]);
            appGeneral(_T("\n"));
        }
        appSafeFree(hostBuffer);
        appPopLogDate();
    }

    static void DebugPrint(
        const CNDeviceTensor<T>& src,
        UINT uiXDim, UINT uiYDim)
    {
        appPushLogDate(FALSE);
        const UINT uiSize = uiXDim * uiYDim;
        T* hostBuffer = (T*)malloc(sizeof(T) * uiSize);
        appGetCuda()->CopyDH(hostBuffer, src.m_pDeviceDataBuffer, sizeof(T) * uiSize);

        appGeneral(_T("{\n"));
        for (UINT x = 0; x < uiXDim; ++x)
        {
            appGeneral(_T("{"));
            for (UINT y = 0; y < uiYDim; ++y)
            {
                LogValue(hostBuffer[x * uiYDim + y]);
                if (y != uiYDim - 1)
                {
                    appGeneral(_T(", "));
                }
            }
            appGeneral(_T("}%s\n"), (x != (uiXDim - 1)) ? _T(",") : _T(""));
        }
        appGeneral(_T("}\n"));
        appSafeFree(hostBuffer);
        appPopLogDate();
    }

    //T* m_pBuffer;

protected:

    /**
     * Things like a = sin(a)
     * Call it like this:
     * TOperator_Sin<FLOAT> op();
     * tensor_common.OneOperator(op, dst, 0, 1, 10, 3)
     *
     * Note, dstStride and lengths are on host
     */
    //template<class Operator>
    //void OneOperator(
    //    const TOperator_D<Operator, T>& op,
    //    T* dst,
    //    const UINT dstIndexStart,
    //    const UINT* __restrict__ dstStride,
    //    const UINT* __restrict__ lengths,
    //    BYTE byIndexCount)
    //{
    //    ((Calculator*)this)->OneOperator(op, dst, dstIndexStart, dstStride, lengths, byIndexCount);
    //}

    /**
     * Things like b = sin(a), where type of b is as same as a
     * Call it like this:
     * TOperator_Sin<FLOAT> op();
     */
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
    //    ((Calculator*)this)->OneOperatorD(op, dst, dstIndexStart, dstStride, 
    //        src, srcIndexStart, srcStride, lengths, byIndexCount);
    //}

#if 0
    /**
     * Things like b = sin(a), where type of b is different than a
     * Call it like this:
     * TOperator_Sin<FLOAT> op();
     */
    template<class Operator, class dstT, class srcT>
    void OneOperatorDS(
        TOperator_DS<Operator, dstT, srcT> op,
        CNDeviceTensor<dstT>* dst,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        CNDeviceTensor<srcT>* src,
        const UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        ((Calculator*)this)->OneOperatorDS(op, dst, dstIndexStart, dstStride,
            src, srcIndexStart, srcStride, lengths, byIndexCount);
    }

    /**
     * Things like c = a + b, type of c is as same as a
     * Call it like this:
     * TOperator_Sin<FLOAT> op();
     * tensor_common.OneOperator(op, dst, 0, 1, 10, 3)
     */
    template<class Operator, class srcTL, class srcTR>
    void TwoOperatorL(
        TOperator_L<Operator, srcTL, srcTR> op,
        CNDeviceTensor<srcTL>* left,
        const UINT leftIndexStart,
        const UINT* __restrict__ leftStride,
        CNDeviceTensor<srcTR>* right,
        const UINT rightIndexStart,
        const UINT* __restrict__ rightStride,
        CNDeviceTensor<srcTL>* dst,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        ((Calculator*)this)->TwoOperatorL(op,
            left, leftIndexStart, leftStride,
            right, rightIndexStart, rightStride,
            dst, dstIndexStart, dstStride, lengths, byIndexCount);
    }

    /**
     * Things like c = a + b, type of c is as same as b
     * Call it like this:
     * TOperator_Sin<FLOAT> op();
     * tensor_common.OneOperator(op, dst, 0, 1, 10, 3)
     */
    template<class Operator, class srcTL, class srcTR>
    void TwoOperatorR(
        TOperator_R<Operator, srcTL, srcTR> op,
        CNDeviceTensor<srcTL>* left,
        const UINT leftIndexStart,
        const UINT* __restrict__ leftStride,
        CNDeviceTensor<srcTR>* right,
        const UINT rightIndexStart,
        const UINT* __restrict__ rightStride,
        CNDeviceTensor<srcTR>* dst,
        const UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        ((Calculator*)this)->TwoOperatorR(op,
            left, leftIndexStart, leftStride,
            right, rightIndexStart, rightStride,
            dst, dstIndexStart, dstStride, lengths, byIndexCount);
    }

    /**
     * Things like c = a + b, type of c is as same as b
     * Call it like this:
     * TOperator_Sin<FLOAT> op();
     * tensor_common.OneOperator(op, dst, 0, 1, 10, 3)
     */
    template<class Operator, class srcTL, class srcTR>
    void TwoOperatorLN(
        TOperator_LN<Operator, srcTL, srcTR> op,
        CNDeviceTensor<srcTL>* left,
        const UINT leftIndexStart,
        const UINT* __restrict__ leftStride,
        CNDeviceTensor<srcTR>* right,
        const UINT rightIndexStart,
        const UINT* __restrict__ rightStride,
        BYTE byIndexCount)
    {
        ((Calculator*)this)->TOperator_LN(op,
            left, leftIndexStart, leftStride,
            right, rightIndexStart, rightStride, byIndexCount);
    }
#endif

    #pragma region functions already implemented



    #pragma endregion
};

template<class Calculator, class Operator, class T>
class __DLL_EXPORT TCNDeviceTensorCommonOneOperator
{
public:
    TCNDeviceTensorCommonOneOperator()
        //: m_pBuffer(pBuffer)
    {

    }

    virtual ~TCNDeviceTensorCommonOneOperator()
    {
        
    }

    virtual void OneOperator(
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount) = 0;

//public:
//
//    T* m_pBuffer;
//    TOperator_D<Operator, T> m_op;
};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_H_

//=============================================================================
// END OF FILE
//=============================================================================