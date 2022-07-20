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
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s) const
    {
        ((Operator*)this)->Do(d, s);
    }

    __inline__ __host__ __device__ dstT Dor(const dstT& d, const srcT& s) const
    {
        dstT r = d;
        ((Operator*)this)->Do(r, s);
        return r;
    }
};

template<class dstT, class srcT>
class __DLL_EXPORT TOperator_Set : public TOperator_S<TOperator_Set<dstT, srcT>, dstT, srcT>
{
public:
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s) const
    {
        d = static_cast<dstT>(s);
    }
};

template<class srcT>
class __DLL_EXPORT TOperator_Set<_SComplex, srcT> : public TOperator_S<TOperator_Set<_SComplex, srcT>, _SComplex, srcT>
{
public:
    __inline__ __host__ __device__ void Do(_SComplex& d, const srcT& s) const
    {
        d = make_cuComplex(static_cast<FLOAT>(s), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT TOperator_Set<dstT, _SComplex> : public TOperator_S<TOperator_Set<dstT, _SComplex>, dstT, _SComplex>
{
public:
    __inline__ __host__ __device__ void Do(dstT& d, const _SComplex& s) const
    {
        d = static_cast<dstT>(cuCabsf(s));
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_SComplex, _DComplex> : public TOperator_S<TOperator_Set<_SComplex, _DComplex>, _SComplex, _DComplex>
{
public:
    __inline__ __host__ __device__ void Do(_SComplex& d, const _DComplex& s) const
    {
        d = make_cuComplex(static_cast<FLOAT>(s.x), static_cast<FLOAT>(s.y));
    }
};

template<class srcT>
class __DLL_EXPORT TOperator_Set<_DComplex, srcT> : public TOperator_S<TOperator_Set<_DComplex, srcT>, _DComplex, srcT>
{
public:
    __inline__ __host__ __device__ void Do(_DComplex& d, const srcT& s) const
    {
        d = make_cuDoubleComplex(static_cast<DOUBLE>(s), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT TOperator_Set<dstT, _DComplex> : public TOperator_S<TOperator_Set<dstT, _DComplex>, dstT, _DComplex>
{
public:
    __inline__ __host__ __device__ void Do(dstT& d, const _DComplex& s) const
    {
        d = static_cast<dstT>(cuCabs(s));
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_DComplex, _SComplex> : public TOperator_S<TOperator_Set<_DComplex, _SComplex>, _DComplex, _SComplex>
{
public:
    __inline__ __host__ __device__ void Do(_DComplex& d, const _SComplex& s) const
    {
        d = make_cuDoubleComplex(static_cast<DOUBLE>(s.x), static_cast<DOUBLE>(s.y));
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_SComplex, _SComplex> : public TOperator_S<TOperator_Set<_SComplex, _SComplex>, _SComplex, _SComplex>
{
public:
    __inline__ __host__ __device__ void Do(_SComplex& d, const _SComplex& s) const
    {
        d = s;
    }
};

template<>
class __DLL_EXPORT TOperator_Set<_DComplex, _DComplex> : public TOperator_S<TOperator_Set<_DComplex, _DComplex>, _DComplex, _DComplex>
{
public:
    __inline__ __host__ __device__ void Do(_DComplex& d, const _DComplex& s) const
    {
        d = s;
    }
};

#pragma endregion

//only dest
#pragma region D

template<class Operator, class dstT>
class __DLL_EXPORT TOperator_D
{
public:
    __inline__ __host__ __device__ void Do(dstT& b) const
    {
        return ((Operator*)this)->Do(b);
    }
};

#define __DEFINE_D_OPERATOR_FUNCTION(funcName) \
template<class dstT> \
class __DLL_EXPORT TOperator_##funcName : public TOperator_D<TOperator_##funcName<dstT>, dstT> \
{ \
public: \
    __inline__ __host__ __device__ void Do(dstT& b) const \
    { \
        b = _##funcName(b); \
    } \
};

__OVER_ALL_ONE_OP(__DEFINE_D_OPERATOR_FUNCTION)

#pragma endregion

//dest - src
#pragma region DS

#define __DEFINE_D_OPERATOR_FUNCTION_S(name) \
template<class dstT, class srcT> \
class __DLL_EXPORT TOperatorS_##name : public TOperator_S<TOperatorS_##name<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s) const \
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
    __inline__ __host__ __device__ void Do(dstT& d, const srcT& s) const \
    { \
        d = _##name(d, s); \
    } \
};

__OVER_ALL_TWO_OP(__DEFINE_D_OPERATOR_FUNCTION_S_TWO)

#pragma endregion


__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_ELEMENTOPERATOR_H_

//=============================================================================
// END OF FILE
//=============================================================================