//=============================================================================
// FILENAME : CNDeviceTensorElementOperator.h
// 
// DESCRIPTION:
// Since function point cannot be template function, we use a template class instead
//
// REVISION:
//  [28/08/2021 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_ELEMENTOPERATOR_H_
#define _CNDEVICETENSOR_ELEMENTOPERATOR_H_

__BEGIN_NAMESPACE

#pragma region One value

/**
 * a=b, a=b*, ...
 */
template<class Calculator, class dstT, class srcT>
class __DLL_EXPORT CElementOperator_DS
{
public:
    __host__ __device__ dstT Do(srcT b)
    {
        return ((Calculator*)this)->Do(b);
    }
};


#pragma region set value

template<class dstT, class srcT>
class __DLL_EXPORT CElementOperator_Set : public CElementOperator_DS<CElementOperator_Set<dstT, srcT>, dstT, srcT>
{
public:
    __host__ __device__ dstT Do(srcT b)
    {
        return static_cast<dstT>(b);
    }
};

template<class srcT>
class __DLL_EXPORT CElementOperator_Set<_SComplex, srcT> : public CElementOperator_DS<CElementOperator_Set<_SComplex, srcT>, _SComplex, srcT>
{
public:
    __host__ __device__ _SComplex Do(srcT b)
    {
        return make_cuComplex(static_cast<FLOAT>(b), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT CElementOperator_Set<dstT, _SComplex> : public CElementOperator_DS<CElementOperator_Set<dstT, _SComplex>, dstT, _SComplex>
{
public:
    __host__ __device__ dstT Do(_SComplex b)
    {
        return static_cast<dstT>(b.x);
    }
};

template<>
class __DLL_EXPORT CElementOperator_Set<_SComplex, _DComplex> : public CElementOperator_DS<CElementOperator_Set<_SComplex, _DComplex>, _SComplex, _DComplex>
{
public:
    __host__ __device__ _SComplex Do(_DComplex b)
    {
        return make_cuComplex(static_cast<FLOAT>(b.x), static_cast<FLOAT>(b.y));
    }
};

template<class srcT>
class __DLL_EXPORT CElementOperator_Set<_DComplex, srcT> : public CElementOperator_DS<CElementOperator_Set<_DComplex, srcT>, _DComplex, srcT>
{
public:
    __host__ __device__ _DComplex Do(srcT b)
    {
        return make_cuDoubleComplex(static_cast<DOUBLE>(b), 0.0f);
    }
};

template<class dstT>
class __DLL_EXPORT CElementOperator_Set<dstT, _DComplex> : public CElementOperator_DS<CElementOperator_Set<dstT, _DComplex>, dstT, _DComplex>
{
public:
    __host__ __device__ dstT Do(_DComplex b)
    {
        return static_cast<dstT>(b.x);
    }
};

template<>
class __DLL_EXPORT CElementOperator_Set<_DComplex, _SComplex> : public CElementOperator_DS<CElementOperator_Set<_DComplex, _SComplex>, _DComplex, _SComplex>
{
public:
    __host__ __device__ _DComplex Do(_SComplex b)
    {
        return make_cuDoubleComplex(static_cast<DOUBLE>(b.x), static_cast<DOUBLE>(b.y));
    }
};

#pragma endregion

#pragma region functions

#define __DEFINE_ONE_OPERATOR_FUNCTION(funcName) \
template<class dstT, class srcT> \
class __DLL_EXPORT CElementOperator##funcName : public CElementOperator_DS<CElementOperator##funcName<dstT, srcT>, dstT, srcT> \
{ \
public: \
    __host__ __device__ dstT Do(srcT b) \
    { \
        CElementOperator_Set<dstT, srcT> setOperator; \
        return setOperator.Do(funcName(b)); \
    } \
};

__DEFINE_ONE_OPERATOR_FUNCTION(_Abs)

__DEFINE_ONE_OPERATOR_FUNCTION(_Re)

__DEFINE_ONE_OPERATOR_FUNCTION(_Im)

__DEFINE_ONE_OPERATOR_FUNCTION(_Conj)

#pragma endregion

#pragma endregion

/**
 * c=a+b, c=a*b, ...
 */
template<class Calculator, class dstT, class srcT1, class srcT2>
class __DLL_EXPORT CElementOperator_DSS
{
public:
    __host__ __device__ dstT Do(srcT1 a, srcT2 b)
    {
        return ((Calculator*)this)->Do(a, b);
    }
};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_ELEMENTOPERATOR_H_

//=============================================================================
// END OF FILE
//=============================================================================