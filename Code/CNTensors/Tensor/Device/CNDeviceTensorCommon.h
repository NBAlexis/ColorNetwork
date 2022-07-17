//=============================================================================
// FILENAME : CNDeviceTensorCommon.h
// 
// DESCRIPTION:
// This is an intereface
//
// REVISION[d-m-y]:
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


#define __Tensor_Common_One_Total(name) \
template<class Tdst> \
void name(Tdst* pBuffer, UINT uiTotalSize) \
{ \
    ((Calc*)this)->name(pBuffer, uiTotalSize); \
}


#define __Tensor_Common_Two_Func_Value_NoReturn(name) \
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

#define __Tensor_Common_Two_Func_Tensor_NoReturn(name) \
template<class Tdst, class Tsrc> \
void name( \
    Tdst* pBuffer, \
    const Tsrc* __restrict__ src, \
    UINT dstIndexStart, \
    const UINT* __restrict__ dstStride, \
    UINT srcIndexStart, \
    const UINT* __restrict__ srcStride, \
    const UINT* __restrict__ lengths, \
    BYTE byIndexCount) \
{ \
    ((Calc*)this)->name(pBuffer, src, dstIndexStart, dstStride, srcIndexStart, srcStride, lengths, byIndexCount); \
}

#define __Tensor_Common_Two_Func_Value_Total(name) \
template<class Tdst, class Tsrc> \
void name(Tdst* pBuffer, const Tsrc& v, UINT uiTotalSize) \
{ \
    ((Calc*)this)->name(pBuffer, v, uiTotalSize); \
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
    void Set(Tdst* pBuffer, const Tsrc& v, UINT uiTotalSize)
    {
        ((Calc*)this)->Set(pBuffer, v, uiTotalSize);
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

    template<class Tdst>
    void Random(Tdst* pBuffer, UINT uiRandomType, UINT uiTotalSize)
    {
        ((Calc*)this)->Random(pBuffer, uiRandomType, uiTotalSize);
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
        ((Calc*)this)->Random(pBuffer, uiRandomType, dstIndexStart, dstStride, lengths, byIndexCount);
    }
    
    __OVER_ALL_ONE_OP(__Tensor_Common_One_Func)
    __OVER_ALL_ONE_OP(__Tensor_Common_One_Total)
    __OVER_ALL_TWO_OP(__Tensor_Common_Two_Func_Value_NoReturn)
    __OVER_ALL_TWO_OP(__Tensor_Common_Two_Func_Tensor_NoReturn)
    __OVER_ALL_TWO_OP(__Tensor_Common_Two_Func_Value_Total)

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
        const UINT uiMaxSize,
        UINT uiXDim, UINT uiYDim)
    {
        appPushLogDate(FALSE);
        if (uiXDim < 1)
        {
            uiXDim = 1;
        }
        if (uiXDim > uiMaxSize)
        {
            uiXDim = uiMaxSize;
        }
        if (uiYDim < 1 || uiYDim * uiXDim > uiMaxSize)
        {
            uiYDim = uiMaxSize / uiXDim;
        }
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

};

//just for calling static functions of TCNDeviceTensorCommon
class __DLL_EXPORT CNDeviceTensorCommonEmpty : public TCNDeviceTensorCommon<CNDeviceTensorCommonEmpty> {};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_COMMON_H_

//=============================================================================
// END OF FILE
//=============================================================================