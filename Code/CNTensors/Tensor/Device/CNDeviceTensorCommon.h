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

//==========================================================
#define __IMPLEMENT_COMMON_ONE1(name, type, calc) \
template class calc<TOperator_##name<type>, type>;

#define __IMPLEMENT_COMMON_ONE2(type, calc) \
__OVER_ALL_ONE_OPAB(__IMPLEMENT_COMMON_ONE1, type, calc)

#define __IMPLEMENT_COMMON_ONE(calc) \
__OVER_ALL_TYPE_ONEA(__IMPLEMENT_COMMON_ONE2, calc)

//==========================================================
#define __IMPLEMENT_COMMON_TWO_SET(type1, type2, name, calc) \
template class calc<TOperator_##name<type1, type2>, type1, type2>;

//==========================================================
#define __IMPLEMENT_COMMON_TWO1(name, type1, type2, calc) \
template class calc<TOperator_##name<type1, type2>, type1, type2>;

#define __IMPLEMENT_COMMON_TWO2(type1, type2, calc) \
__OVER_ALL_TWO_OPABC(__IMPLEMENT_COMMON_TWO1, type1, type2, calc)

#define __IMPLEMENT_COMMON_TWO(calc) \
__OVER_ALL_TYPE_TWOA(__IMPLEMENT_COMMON_TWO2, calc)


#define __Tensor_Common_One_Func(name) \
template<class T> \
void name(T* pBuffer, UINT dstIndexStart, const UINT* __restrict__ dstStride, const UINT* __restrict__ lengths, BYTE byIndexCount) \
{ \
    ((Calc*)this)->name(pBuffer, dstIndexStart, dstStride, lengths, byIndexCount); \
}

#define __Tensor_Common_Two_Func_NoReturn(name) \
template<class Tdst, class Tsrc> \
void name( \
    Tdst* pBuffer, \
    const Tsrc& v, \
    UINT dstIndexStart, \
    const UINT* __restrict__ dstStride, \
    const UINT* __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    ((Calc*)this)->name(pBuffer, v, dstIndexStart, dstStride, lengths, byIndexCount); \
}

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
template<class Calc>
class __DLL_EXPORT TCNDeviceTensorCommon
{
public:

    virtual ~TCNDeviceTensorCommon()
    {
        
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
        ((Calc*)this)->Set(pBuffer, v, dstIndexStart, dstStride, lengths, byIndexCount);
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
        ((Calc*)this)->Set(pBuffer, src, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount);
    }

    __OVER_ALL_ONE_OP(__Tensor_Common_One_Func)
    __OVER_ALL_TWO_OP(__Tensor_Common_Two_Func_NoReturn)

    template<class Tdst, class Tsrc>
    void Add(
        Tdst* pBuffer,
        const Tsrc* __restrict__ src,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount)
    {
        ((Calc*)this)->Add(pBuffer, src, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount);
    }

    template<class T>
    static void DebugPrint(const T* __restrict__ src, UINT uiSize)
    {
        appPushLogDate(FALSE);
        T* hostBuffer = (T*)malloc(sizeof(T) * uiSize);
        appGetCuda()->CopyDH(hostBuffer, src, sizeof(T) * uiSize);
        for (UINT i = 0; i < uiSize; ++i)
        {
            appGeneral(_T("%d: "), i);
            LogValue(hostBuffer[i]);
            appGeneral(_T("\n"));
        }
        appSafeFree(hostBuffer);
        appPopLogDate();
    }

    template<class T>
    static void DebugPrint(
        const T* __restrict__ src,
        UINT uiXDim, UINT uiYDim)
    {
        appPushLogDate(FALSE);
        const UINT uiSize = uiXDim * uiYDim;
        T* hostBuffer = (T*)malloc(sizeof(T) * uiSize);
        appGetCuda()->CopyDH(hostBuffer, src, sizeof(T) * uiSize);

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

class __DLL_EXPORT CNDeviceTensorCommonEmpty : public TCNDeviceTensorCommon<CNDeviceTensorCommonEmpty>
{
};

template<class Calculator, class Operator, class T>
class __DLL_EXPORT TCNDeviceTensorCommonOneOperator
{
public:
    TCNDeviceTensorCommonOneOperator() { }
    virtual ~TCNDeviceTensorCommonOneOperator() { }
    virtual void OneOperator(T* pBuffer, UINT dstIndexStart, const UINT* __restrict__ dstStride, const UINT* __restrict__ lengths, BYTE byIndexCount) = 0;
};

template<class Calculator, class Operator, class Tdst, class Tsrc>
class __DLL_EXPORT TCNDeviceTensorCommonTwoOperator
{
public:
    TCNDeviceTensorCommonTwoOperator()
    {

    }

    virtual ~TCNDeviceTensorCommonTwoOperator()
    {

    }

    virtual void TwoOperatorValue(
        Tdst* pBuffer,
        const Tsrc& v,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount) = 0;

    virtual void TwoOperatorTensor(
        Tdst* pBuffer,
        const Tsrc* __restrict__ src,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        UINT srcIndexStart,
        const UINT* __restrict__ srcStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount) = 0;
};

template<class Calculator, class Operator, class Tdst, class Tsrc>
class __DLL_EXPORT TCNDeviceTensorCommonTwoOperatorDDS
{
public:
    TCNDeviceTensorCommonTwoOperatorDDS()
    {

    }

    virtual ~TCNDeviceTensorCommonTwoOperatorDDS()
    {

    }

    virtual void TwoOperator(
        Tdst* pBuffer,
        const Tsrc& v,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount) = 0;

};

template<class Calculator, class Operator, class Tdst, class Tsrc>
class __DLL_EXPORT TCNDeviceTensorCommonTwoOperatorDSS
{
public:
    TCNDeviceTensorCommonTwoOperatorDSS()
    {

    }

    virtual ~TCNDeviceTensorCommonTwoOperatorDSS()
    {

    }

    virtual void TwoOperator(
        Tdst* pBuffer,
        const Tsrc& v,
        UINT dstIndexStart,
        const UINT* __restrict__ dstStride,
        const UINT* __restrict__ lengths,
        BYTE byIndexCount) = 0;

};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_H_

//=============================================================================
// END OF FILE
//=============================================================================