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

#pragma region set value

template<class Operator, class dstT, class srcT>
class __DLL_EXPORT TOperator_S
{
public:
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s)
    {
        ((Operator*)this)->Do(d, s);
    }
};

template<class dstT, class srcT>
class __DLL_EXPORT TOperator_Set : public TOperator_S<TOperator_Set<dstT, srcT>, dstT, srcT>
{
public:
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s)
    {
        d = static_cast<dstT>(s);
    }
};

template<class srcT>
class __DLL_EXPORT TOperator_Set<_SComplex, srcT> : public TOperator_S<TOperator_Set<_SComplex, srcT>, _SComplex, srcT>
{
public:
    __inline__ __host__ __device__ void Do(_SComplex& d, const srcT& s)
    {
        d = make_cuComplex(static_cast<FLOAT>(s), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT TOperator_Set<dstT, _SComplex> : public TOperator_S<TOperator_Set<dstT, _SComplex>, dstT, _SComplex>
{
public:
    __inline__ __host__ __device__ void Do(dstT& d, const _SComplex& s)
    {
        d = static_cast<dstT>(cuCabsf(s));
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_SComplex, _DComplex> : public TOperator_S<TOperator_Set<_SComplex, _DComplex>, _SComplex, _DComplex>
{
public:
    __inline__ __host__ __device__ void Do(_SComplex& d, const _DComplex& s)
    {
        d = make_cuComplex(static_cast<FLOAT>(s.x), static_cast<FLOAT>(s.y));
    }
};

template<class srcT>
class __DLL_EXPORT TOperator_Set<_DComplex, srcT> : public TOperator_S<TOperator_Set<_DComplex, srcT>, _DComplex, srcT>
{
public:
    __inline__ __host__ __device__ void Do(_DComplex& d, const srcT& s)
    {
        d = make_cuDoubleComplex(static_cast<DOUBLE>(s), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT TOperator_Set<dstT, _DComplex> : public TOperator_S<TOperator_Set<dstT, _DComplex>, dstT, _DComplex>
{
public:
    __inline__ __host__ __device__ void Do(dstT& d, const _DComplex& s)
    {
        d = static_cast<dstT>(cuCabs(s));
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_DComplex, _SComplex> : public TOperator_S<TOperator_Set<_DComplex, _SComplex>, _DComplex, _SComplex>
{
public:
    __inline__ __host__ __device__ void Do(_DComplex& d, const _SComplex& s)
    {
        d = make_cuDoubleComplex(static_cast<DOUBLE>(s.x), static_cast<DOUBLE>(s.y));
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_SComplex, _SComplex> : public TOperator_S<TOperator_Set<_SComplex, _SComplex>, _SComplex, _SComplex>
{
public:
    __inline__ __host__ __device__ void Do(_SComplex& d, const _SComplex& s)
    {
        d = s;
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_DComplex, _DComplex> : public TOperator_S<TOperator_Set<_DComplex, _DComplex>, _DComplex, _DComplex>
{
public:
    __inline__ __host__ __device__ void Do(_DComplex& d, const _DComplex& s)
    {
        d = s;
    }
};

#pragma endregion

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

#define __DEFINE_D_OPERATOR_FUNCTION(funcName) \
template<class dstT> \
class __DLL_EXPORT TOperator_##funcName : public TOperator_D<TOperator_##funcName<dstT>, dstT> \
{ \
public: \
    __inline__ __host__ __device__ void Do(dstT& b) \
    { \
        b = _##funcName(b); \
    } \
};

__OVER_ALL_ONE_OP(__DEFINE_D_OPERATOR_FUNCTION)

#pragma endregion

#pragma region dest - src



#pragma region functions


//#define __DEFINE_DS_OPERATOR_FUNCTION(funcName) \
//template<class dstT, class srcT> \
//class __DLL_EXPORT TOperatorDS_##funcName : public TOperator_DS<TOperatorDS_##funcName<dstT, srcT>, dstT, srcT> \
//{ \
//public: \
//    __inline__ __host__ __device__ dstT Do(const srcT& b) \
//    { \
//        TOperator_Set<dstT, srcT> setOperator; \
//        return setOperator.Do(_##funcName(b)); \
//    } \
//};
//
//__OVER_ALL_ONE_OP(__DEFINE_DS_OPERATOR_FUNCTION)


#define __DEFINE_D_OPERATOR_FUNCTION_S(name) \
template<class dstT, class srcT> \
class __DLL_EXPORT TOperatorS_##name : public TOperator_S<TOperatorS_##name<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s) \
    { \
        TOperator_Set<dstT, srcT> setOperator; \
        setOperator.Do(d, _##name(s)); \
    } \
};

__OVER_ALL_ONE_OP(__DEFINE_D_OPERATOR_FUNCTION_S)

#define __DEFINE_D_OPERATOR_FUNCTION_S_TWO(name) \
template<class dstT, class srcT> \
class __DLL_EXPORT TOperator_##name : public TOperator_S<TOperator_##name<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s) \
    { \
        d = _##name(d, s); \
    } \
};

__OVER_ALL_TWO_OP(__DEFINE_D_OPERATOR_FUNCTION_S_TWO)

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
class __DLL_EXPORT TOperatorL_##funcName : public TOperator_L<TOperatorL_##funcName<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ dstT Do(dstT a, srcT b) \
    { \
        return _##funcName(a, b); \
    } \
};

__OVER_ALL_TWO_OP(__DEFINE_DSS_OPERATOR_FUNCTION_L)


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