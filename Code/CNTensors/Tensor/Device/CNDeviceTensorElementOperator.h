//=============================================================================
// FILENAME : CNDeviceTensorElementOperator.h
// 
// DESCRIPTION:
// Since function point cannot be template function, we use a template class instead
// Note that:
// https://stackoverflow.com/questions/2354210/can-a-class-member-function-template-be-virtual#
// there are two usable approaches:
// use a virtual function table with specified types (this can be simplified using MACROs)
// use "CRTP"
// the inconvenience of CRTP is that we have to keep the function template always
// NOTE that we have to pass the object to kernel functions.
// That will make the kernel functions template functions.
// Note that CRTP can make the calls __inline__
//
// REVISION:
//  [28/08/2021 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_ELEMENTOPERATOR_H_
#define _CNDEVICETENSOR_ELEMENTOPERATOR_H_

__BEGIN_NAMESPACE

#pragma region only dest

template<class Operator, class dstT>
class __DLL_EXPORT TOperator_D
{
public:
    __inline__ __host__ __device__ void Do(dstT& b)
    {
        return ((Operator*)this)->Do(b);
    }
};

#define __DEFINE_D_OPERATOR_FUNCTION_NORETURN(funcName) \
template<class dstT> \
class __DLL_EXPORT TOperator##funcName : public TOperator_D<TOperator##funcName<dstT>, dstT> \
{ \
public: \
    __inline__ __host__ __device__ void Do(dstT& b) \
    { \
        funcName(b); \
    } \
};

#define __DEFINE_D_OPERATOR_FUNCTION_RETURN(funcName) \
template<class dstT> \
class __DLL_EXPORT TOperator##funcName : public TOperator_D<TOperator##funcName<dstT>, dstT> \
{ \
public: \
    __inline__ __host__ __device__ void Do(dstT& b) \
    { \
        b = funcName(b); \
    } \
};

__DEFINE_D_OPERATOR_FUNCTION_NORETURN(_Zero)

__DEFINE_D_OPERATOR_FUNCTION_NORETURN(_One)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Abs)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Re)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Im)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Conj)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Arg)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Pow)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Exp)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Log)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Sqrt)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Sin)

__DEFINE_D_OPERATOR_FUNCTION_RETURN(_Cos)

#pragma endregion

#pragma region dest - src

/**
 * a=b, a=b*, ...
 */
template<class Operator, class dstT, class srcT>
class __DLL_EXPORT TOperator_DS
{
public:
    __inline__ __host__ __device__ dstT Do(srcT b)
    {
        return ((Operator*)this)->Do(b);
    }
};


#pragma region set value

template<class dstT, class srcT>
class __DLL_EXPORT TOperator_Set : public TOperator_DS<TOperator_Set<dstT, srcT>, dstT, srcT>
{
public:
    __inline__ __host__ __device__ dstT Do(srcT b)
    {
        return static_cast<dstT>(b);
    }
};

template<class srcT>
class __DLL_EXPORT TOperator_Set<_SComplex, srcT> : public TOperator_DS<TOperator_Set<_SComplex, srcT>, _SComplex, srcT>
{
public:
    __inline__ __host__ __device__ _SComplex Do(srcT b)
    {
        return make_cuComplex(static_cast<FLOAT>(b), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT TOperator_Set<dstT, _SComplex> : public TOperator_DS<TOperator_Set<dstT, _SComplex>, dstT, _SComplex>
{
public:
    __inline__ __host__ __device__ dstT Do(_SComplex b)
    {
        return static_cast<dstT>(b.x);
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_SComplex, _DComplex> : public TOperator_DS<TOperator_Set<_SComplex, _DComplex>, _SComplex, _DComplex>
{
public:
    __inline__ __host__ __device__ _SComplex Do(_DComplex b)
    {
        return make_cuComplex(static_cast<FLOAT>(b.x), static_cast<FLOAT>(b.y));
    }
};

template<class srcT>
class __DLL_EXPORT TOperator_Set<_DComplex, srcT> : public TOperator_DS<TOperator_Set<_DComplex, srcT>, _DComplex, srcT>
{
public:
    __inline__ __host__ __device__ _DComplex Do(srcT b)
    {
        return make_cuDoubleComplex(static_cast<DOUBLE>(b), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT TOperator_Set<dstT, _DComplex> : public TOperator_DS<TOperator_Set<dstT, _DComplex>, dstT, _DComplex>
{
public:
    __inline__ __host__ __device__ dstT Do(_DComplex b)
    {
        return static_cast<dstT>(b.x);
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_DComplex, _SComplex> : public TOperator_DS<TOperator_Set<_DComplex, _SComplex>, _DComplex, _SComplex>
{
public:
    __inline__ __host__ __device__ _DComplex Do(_SComplex b)
    {
        return make_cuDoubleComplex(static_cast<DOUBLE>(b.x), static_cast<DOUBLE>(b.y));
    }
};

#pragma endregion

#pragma region functions

#define __DEFINE_DS_OPERATOR_FUNCTION(funcName) \
template<class dstT, class srcT> \
class __DLL_EXPORT TOperatorDS##funcName : public TOperator_DS<TOperatorDS##funcName<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ dstT Do(srcT b) \
    { \
        TOperator_Set<dstT, srcT> setOperator; \
        return setOperator.Do(funcName(b)); \
    } \
};

__DEFINE_DS_OPERATOR_FUNCTION(_Abs)

__DEFINE_DS_OPERATOR_FUNCTION(_Re)

__DEFINE_DS_OPERATOR_FUNCTION(_Im)

__DEFINE_DS_OPERATOR_FUNCTION(_Conj)

__DEFINE_DS_OPERATOR_FUNCTION(_Arg)

__DEFINE_DS_OPERATOR_FUNCTION(_Pow)

__DEFINE_DS_OPERATOR_FUNCTION(_Exp)

__DEFINE_DS_OPERATOR_FUNCTION(_Log)

__DEFINE_DS_OPERATOR_FUNCTION(_Sqrt)

__DEFINE_DS_OPERATOR_FUNCTION(_Sin)

__DEFINE_DS_OPERATOR_FUNCTION(_Cos)

#pragma endregion

#pragma endregion

#pragma region dest src1 src2

/**
 * c=a+b, c=a*b, ...
 * with type same as a
 */
template<class Calculator, class dstT, class srcT>
class __DLL_EXPORT TOperator_L
{
public:
    __inline__ __host__ __device__ dstT Do(dstT a, srcT b)
    {
        return ((Calculator*)this)->Do(a, b);
    }
};

#define __DEFINE_DSS_OPERATOR_FUNCTION_L(funcName) \
template<class dstT, class srcT> \
class __DLL_EXPORT TOperatorL##funcName : public TOperator_L<TOperatorL##funcName<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ dstT Do(dstT a, srcT b) \
    { \
        return funcName(a, b); \
    } \
};

__DEFINE_DSS_OPERATOR_FUNCTION_L(_Add)

__DEFINE_DSS_OPERATOR_FUNCTION_L(_Mul)

__DEFINE_DSS_OPERATOR_FUNCTION_L(_Sub)

__DEFINE_DSS_OPERATOR_FUNCTION_L(_Div)

/**
 * c=a+b, c=a*b, ...
 * type same as b
 */
template<class Calculator, class dstT, class srcT>
class __DLL_EXPORT TOperator_R
{
public:
    __inline__ __host__ __device__ dstT Do(srcT a, dstT b)
    {
        return ((Calculator*)this)->Do(a, b);
    }
};

#define __DEFINE_DSS_OPERATOR_FUNCTION_R(funcName) \
template<class dstT, class srcT> \
class __DLL_EXPORT TOperatorR##funcName : public TOperator_R<TOperatorR##funcName<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ dstT Do(srcT a, dstT b) \
    { \
        return funcName(a, b); \
    } \
};

__DEFINE_DSS_OPERATOR_FUNCTION_R(_Sub_R)

__DEFINE_DSS_OPERATOR_FUNCTION_R(_Div_R)


/**
 * a=a+b, a=a*b, ...
  */
template<class Calculator, class dstT, class srcT>
class __DLL_EXPORT TOperator_LN
{
public:
    __inline__ __host__ __device__ void Do(dstT& a, srcT b)
    {
        a = ((Calculator*)this)->Do(a, b);
    }
};


#define __DEFINE_DSS_OPERATOR_FUNCTION_LN(funcName) \
template<class dstT, class srcT> \
class __DLL_EXPORT TOperatorLN##funcName : public TOperator_LN<TOperatorLN##funcName<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ dstT Do(dstT a, srcT b) \
    { \
        a = funcName(a, b); \
    } \
};

__DEFINE_DSS_OPERATOR_FUNCTION_LN(_Add)

__DEFINE_DSS_OPERATOR_FUNCTION_LN(_Mul)

__DEFINE_DSS_OPERATOR_FUNCTION_LN(_Sub)

__DEFINE_DSS_OPERATOR_FUNCTION_LN(_Div)


#pragma endregion

#pragma region dest src1 src2 src3

/**
 * c = a*x + y, c = a*(x + y), ...
 */
template<class Calculator, class dstT, class srcT1, class srcT2, class srcT3>
class __DLL_EXPORT CElementOperator_DSSS
{
public:
    __inline__ __host__ __device__ dstT Do(srcT1 a, srcT2 b)
    {
        return ((Calculator*)this)->Do(a, b);
    }
};

#pragma endregion

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_ELEMENTOPERATOR_H_

//=============================================================================
// END OF FILE
//=============================================================================