//=============================================================================
// FILENAME : CudaComplexFunction.h
// 
// DESCRIPTION:
// Add some function to CNComplex where it does not have yet
//
//
// REVISION:
//  [01/06/2020 nbale]
//=============================================================================

#ifndef _CUDACOMPLEXFUNCTION_H_
#define _CUDACOMPLEXFUNCTION_H_

/**
* Do not use _VA_ARGS_
* for example:
#define AA(mac, a, ...) mac(a, __VA_ARGS__)
#define MAC1(a, b, c) a##b##c

AA(MAC1, 0, 1, 2)
is:
01,2
* 
*/

#define __OVER_ALL_ONE_OP(mac) \
mac(Zero) \
mac(One) \
mac(Abs) \
mac(Re) \
mac(Im) \
mac(AbsSq) \
mac(Conj) \
mac(Arg) \
mac(Exp) \
mac(Log) \
mac(Sqrt) \
mac(Sin) \
mac(Cos) \
mac(Oppo) \
mac(Inv) 

#define __OVER_ALL_TWO_OP(mac) \
mac(Add) \
mac(Mul) \
mac(Sub) \
mac(Div) \
mac(Pow) \
mac(SubR) \
mac(DivR) 


#define __OVER_ALL_TYPE_ONE(mac) \
mac(SBYTE) \
mac(INT) \
mac(FLOAT) \
mac(DOUBLE) \
mac(_SComplex) \
mac(_DComplex) 


#define __OVER_ALL_ONE_OPA(mac, a) \
mac(Zero, a) \
mac(One, a) \
mac(Abs, a) \
mac(Re, a) \
mac(Im, a) \
mac(AbsSq, a) \
mac(Conj, a) \
mac(Arg, a) \
mac(Exp, a) \
mac(Log, a) \
mac(Sqrt, a) \
mac(Sin, a) \
mac(Cos, a) \
mac(Oppo, a) \
mac(Inv, a) 

#define __OVER_ALL_TWO_OPA(mac, a) \
mac(Add, a) \
mac(Mul, a) \
mac(Sub, a) \
mac(Div, a) \
mac(Pow, a) \
mac(SubR, a) \
mac(DivR, a) 


#define __OVER_ALL_TYPE_ONEA(mac, a) \
mac(SBYTE, a) \
mac(INT, a) \
mac(FLOAT, a) \
mac(DOUBLE, a) \
mac(_SComplex, a) \
mac(_DComplex, a) 


#define __OVER_ALL_ONE_OPAB(mac, a, b) \
mac(Zero, a, b) \
mac(One, a, b) \
mac(Abs, a, b) \
mac(Re, a, b) \
mac(Im, a, b) \
mac(AbsSq, a, b) \
mac(Conj, a, b) \
mac(Arg, a, b) \
mac(Exp, a, b) \
mac(Log, a, b) \
mac(Sqrt, a, b) \
mac(Sin, a, b) \
mac(Cos, a, b) \
mac(Oppo, a, b) \
mac(Inv, a, b) 

#define __OVER_ALL_TWO_OPAB(mac, a, b) \
mac(Add, a, b) \
mac(Mul, a, b) \
mac(Sub, a, b) \
mac(Div, a, b) \
mac(Pow, a, b) \
mac(SubR, a, b) \
mac(DivR, a, b) 


#define __OVER_ALL_TYPE_ONEAB(mac, a, b) \
mac(SBYTE, a, b) \
mac(INT, a, b) \
mac(FLOAT, a, b) \
mac(DOUBLE, a, b) \
mac(_SComplex, a, b) \
mac(_DComplex, a, b) 


#define __OVER_ALL_ONE_OPABC(mac, a, b, c) \
mac(Zero, a, b, c) \
mac(One, a, b, c) \
mac(Abs, a, b, c) \
mac(Re, a, b, c) \
mac(Im, a, b, c) \
mac(AbsSq, a, b, c) \
mac(Conj, a, b, c) \
mac(Arg, a, b, c) \
mac(Exp, a, b, c) \
mac(Log, a, b, c) \
mac(Sqrt, a, b, c) \
mac(Sin, a, b, c) \
mac(Cos, a, b, c) \
mac(Oppo, a, b, c) \
mac(Inv, a, b, c) 

#define __OVER_ALL_TWO_OPABC(mac, a, b, c) \
mac(Add, a, b, c) \
mac(Mul, a, b, c) \
mac(Sub, a, b, c) \
mac(Div, a, b, c) \
mac(Pow, a, b, c) \
mac(SubR, a, b, c) \
mac(DivR, a, b, c) 


#define __OVER_ALL_TYPE_ONEABC(mac, a, b, c) \
mac(SBYTE, a, b, c) \
mac(INT, a, b, c) \
mac(FLOAT, a, b, c) \
mac(DOUBLE, a, b, c) \
mac(_SComplex, a, b, c) \
mac(_DComplex, a, b, c) 

#define __OVER_ALL_TYPE_TWO_LEVEL1(mac, type2) \
mac(SBYTE, type2) \
mac(INT, type2) \
mac(FLOAT, type2) \
mac(DOUBLE, type2) \
mac(_SComplex, type2) \
mac(_DComplex, type2) 


#define __OVER_ALL_TYPE_TWO_LEVEL1A(mac, type2, a) \
mac(SBYTE, type2, a) \
mac(INT, type2, a) \
mac(FLOAT, type2, a) \
mac(DOUBLE, type2, a) \
mac(_SComplex, type2, a) \
mac(_DComplex, type2, a) 


#define __OVER_ALL_TYPE_TWO_LEVEL1AB(mac, type2, a, b) \
mac(SBYTE, type2, a, b) \
mac(INT, type2, a, b) \
mac(FLOAT, type2, a, b) \
mac(DOUBLE, type2, a, b) \
mac(_SComplex, type2, a, b) \
mac(_DComplex, type2, a, b) 

#define __OVER_ALL_TYPE_TWO_LEVEL1ABC(mac, type2, a, b, c) \
mac(SBYTE, type2, a, b, c) \
mac(INT, type2, a, b, c) \
mac(FLOAT, type2, a, b, c) \
mac(DOUBLE, type2, a, b, c) \
mac(_SComplex, type2, a, b, c) \
mac(_DComplex, type2, a, b, c) 

#define __OVER_ALL_TYPE_TWO(mac) \
__OVER_ALL_TYPE_TWO_LEVEL1(mac, SBYTE) \
__OVER_ALL_TYPE_TWO_LEVEL1(mac, INT) \
__OVER_ALL_TYPE_TWO_LEVEL1(mac, FLOAT) \
__OVER_ALL_TYPE_TWO_LEVEL1(mac, DOUBLE) \
__OVER_ALL_TYPE_TWO_LEVEL1(mac, _SComplex) \
__OVER_ALL_TYPE_TWO_LEVEL1(mac, _DComplex) 


#define __OVER_ALL_TYPE_TWOA(mac, a) \
__OVER_ALL_TYPE_TWO_LEVEL1A(mac, SBYTE, a) \
__OVER_ALL_TYPE_TWO_LEVEL1A(mac, INT, a) \
__OVER_ALL_TYPE_TWO_LEVEL1A(mac, FLOAT, a) \
__OVER_ALL_TYPE_TWO_LEVEL1A(mac, DOUBLE, a) \
__OVER_ALL_TYPE_TWO_LEVEL1A(mac, _SComplex, a) \
__OVER_ALL_TYPE_TWO_LEVEL1A(mac, _DComplex, a)


#define __OVER_ALL_TYPE_TWOAB(mac, a, b) \
__OVER_ALL_TYPE_TWO_LEVEL1AB(mac, SBYTE, a, b) \
__OVER_ALL_TYPE_TWO_LEVEL1AB(mac, INT, a, b) \
__OVER_ALL_TYPE_TWO_LEVEL1AB(mac, FLOAT, a, b) \
__OVER_ALL_TYPE_TWO_LEVEL1AB(mac, DOUBLE, a, b) \
__OVER_ALL_TYPE_TWO_LEVEL1AB(mac, _SComplex, a, b) \
__OVER_ALL_TYPE_TWO_LEVEL1AB(mac, _DComplex, a, b)


#define __OVER_ALL_TYPE_TWOABC(mac, a, b, c) \
__OVER_ALL_TYPE_TWO_LEVEL1ABC(mac, SBYTE, a, b, c) \
__OVER_ALL_TYPE_TWO_LEVEL1ABC(mac, INT, a, b, c) \
__OVER_ALL_TYPE_TWO_LEVEL1ABC(mac, FLOAT, a, b, c) \
__OVER_ALL_TYPE_TWO_LEVEL1ABC(mac, DOUBLE, a, b, c) \
__OVER_ALL_TYPE_TWO_LEVEL1ABC(mac, _SComplex, a, b, c) \
__OVER_ALL_TYPE_TWO_LEVEL1ABC(mac, _DComplex, a, b, c)

__BEGIN_NAMESPACE

#pragma region Device Math Function

#pragma endregion

#pragma region One element ops

template<class T> __inline__ __host__ __device__
T _Zero(const T& ) { return static_cast<T>(0); }

template<> __inline__ __host__ __device__
_SComplex _Zero(const _SComplex& a) { return _zerocs; }

template<> __inline__ __host__ __device__
_DComplex _Zero(const _DComplex& a) { return _zerocd; }

template<class T> __inline__ __host__ __device__
T _One(const T& ) { return static_cast<T>(1); }

template<> __inline__ __host__ __device__
_SComplex _One(const _SComplex& ) { return _onecs; }

template<> __inline__ __host__ __device__
_DComplex _One(const _DComplex& ) { return _onecd; }

template<class T> __inline__ __host__ __device__
T _Abs(const T& a) { return a >= 0 ? a : (-a); }

template<> __inline__ __host__ __device__
BYTE _Abs(const BYTE& a) { return a; }

template<> __inline__ __host__ __device__
USHORT _Abs(const USHORT& a) { return a; }

template<> __inline__ __host__ __device__
UINT _Abs(const UINT& a) { return a; }

template<> __inline__ __host__ __device__
QWORD _Abs(const QWORD& a) { return a; }

template<> __inline__ __host__ __device__
_SComplex _Abs(const _SComplex& a) { return make_cuComplex(cuCabsf(a), 0.0f); }

template<> __inline__ __host__ __device__
_DComplex _Abs(const _DComplex& a) { return make_cuDoubleComplex(cuCabs(a), 0.0); }

template<class T> __inline__ __host__ __device__
T _Re(const T& a) { return a; }

template<> __inline__ __host__ __device__
_SComplex _Re(const _SComplex& a) { return make_cuComplex(a.x, 0.0f); }

template<> __inline__ __host__ __device__
_DComplex _Re(const _DComplex& a) { return make_cuDoubleComplex(a.x, 0.0); }

template<class T> __inline__ __host__ __device__
T _Im(const T& a) { return static_cast<T>(0); }

template<> __inline__ __host__ __device__
_SComplex _Im(const _SComplex& a) { return make_cuComplex(a.y, 0.0f); }

template<> __inline__ __host__ __device__
_DComplex _Im(const _DComplex& a) { return make_cuDoubleComplex(a.y, 0.0); }

template<class T> __inline__ __host__ __device__
T _AbsSq(const T& a) { return a * a; }

__inline__ __host__ __device__
_SComplex _AbsSq(const _SComplex& a) { return make_cuComplex(a.x * a.x + a.y * a.y, 0.0f); }

__inline__ __host__ __device__
_DComplex _AbsSq(const _DComplex& a) { return make_cuDoubleComplex(a.x * a.x + a.y * a.y, 0.0); }

template<class T> __inline__ __host__ __device__
T _Conj(const T& a) { return a; }

template<> __inline__ __host__ __device__
_SComplex _Conj(const _SComplex& a) { return cuConjf(a); }

template<> __inline__ __host__ __device__
_DComplex _Conj(const _DComplex& a) { return cuConj(a); }

template<class T> __inline__ __host__ __device__
constexpr T _Arg(const T& a) { return static_cast<T>(0); }

__inline__ __host__ __device__
_SComplex _Arg(const _SComplex& a) { return make_cuComplex(atan2(cuCimagf(a), cuCrealf(a)), 0.0f); }

__inline__ __host__ __device__
_DComplex _Arg(const _DComplex& a) { return make_cuDoubleComplex(atan2(cuCimag(a), cuCreal(a)), 0.0); }

template<class T> __inline__ __host__ __device__
T _Exp(const T& a) { return static_cast<T>(exp(static_cast<FLOAT>(a))); }

template<> __inline__ __host__ __device__
DOUBLE _Exp(const DOUBLE& a)
{
    return exp(a);
}

template<> __inline__ __host__ __device__
_SComplex _Exp(const _SComplex& a)
{
    const FLOAT factor = exp(a.x);
    return make_cuComplex(factor * cos(a.y), factor * sin(a.y));
}

template<> __inline__ __host__ __device__
_DComplex _Exp(const _DComplex& a)
{
    const DOUBLE factor = exp(a.x);
    return make_cuDoubleComplex(factor * cos(a.y), factor * sin(a.y));
}

template<class T> __inline__ __host__ __device__
T _Log(const T& a) { return static_cast<T>(log(static_cast<FLOAT>(a))); }

template<> __inline__ __host__ __device__
DOUBLE _Log(const DOUBLE& a) { return log(a); }

template<> __inline__ __host__ __device__
_SComplex _Log(const _SComplex& a)
{
    const FLOAT fArg = atan2(cuCimagf(a), cuCrealf(a));
    return make_cuComplex(log(cuCabsf(a)), fArg > PIF ? fArg - PI2F : fArg);
}

template<> __inline__ __host__ __device__
_DComplex _Log(const _DComplex& a)
{
    const DOUBLE fArg = atan2(cuCimag(a), cuCreal(a));
    return make_cuDoubleComplex(log(cuCabs(a)), fArg > PI ? fArg - PI2 : fArg);
}

template<class T> __inline__ __host__ __device__
T _Sqrt(const T& a) { return static_cast<T>(sqrt(static_cast<FLOAT>(a))); }

template<> __inline__ __host__ __device__
DOUBLE _Sqrt(const DOUBLE& a) { return sqrt(a); }

template<> __inline__ __host__ __device__
_SComplex _Sqrt(const _SComplex& a)
{
    const FLOAT fRadius = cuCabsf(a);
    const FLOAT fCosA = a.x / fRadius;
    _SComplex out;
    out.x = sqrt(0.5f * fRadius * (fCosA + 1.0f));
    out.y = sqrt(0.5f * fRadius * (1.0f - fCosA));
    if (a.y < 0.0f)
        out.y *= -1.0f;

    return out;
}

template<> __inline__ __host__ __device__
_DComplex _Sqrt(const _DComplex& a)
{
    const DOUBLE fRadius = cuCabs(a);
    const DOUBLE fCosA = a.x / fRadius;
    _DComplex out;
    out.x = sqrt(0.5 * fRadius * (fCosA + 1.0));
    out.y = sqrt(0.5 * fRadius * (1.0 - fCosA));
    if (a.y < 0.0)
        out.y *= -1.0;

    return out;
}

template<class T> __inline__ __host__ __device__
T _Sin(const T& a) { return static_cast<T>(sin(static_cast<FLOAT>(a))); }

template<> __inline__ __host__ __device__
DOUBLE _Sin(const DOUBLE &a) { return sin(a); }

template<> __inline__ __host__ __device__
_SComplex _Sin(const _SComplex& a)
{
    const _SComplex numerator = cuCsubf(
        _Exp(make_cuComplex(-a.y, a.x)), 
        _Exp(make_cuComplex(a.y, -a.x)));

    return make_cuComplex(numerator.y * 0.5f, numerator.x * -0.5f);
}

template<> __inline__ __host__ __device__
_DComplex _Sin(const _DComplex& a)
{
    const _DComplex numerator = cuCsub(
        _Exp(make_cuDoubleComplex(-a.y, a.x)),
        _Exp(make_cuDoubleComplex(a.y, -a.x)));

    return make_cuDoubleComplex(numerator.y * 0.5, numerator.x * -0.5);
}

template<class T> __inline__ __host__ __device__
T _Cos(const T& a) { return static_cast<T>(cos(static_cast<FLOAT>(a))); }

template<> __inline__ __host__ __device__
DOUBLE _Cos(const DOUBLE& a) { return cos(a); }

template<> __inline__ __host__ __device__
_SComplex _Cos(const _SComplex& a)
{
    const _SComplex numerator = cuCaddf(
        _Exp(make_cuComplex(-a.y, a.x)),
        _Exp(make_cuComplex(a.y, -a.x)));

    return make_cuComplex(numerator.x * 0.5f, numerator.y * 0.5f);
}

template<> __inline__ __host__ __device__
_DComplex _Cos(const _DComplex& a)
{
    const _DComplex numerator = cuCadd(
        _Exp(make_cuDoubleComplex(-a.y, a.x)),
        _Exp(make_cuDoubleComplex(a.y, -a.x)));

    return make_cuDoubleComplex(numerator.x * 0.5, numerator.y * 0.5);
}

template<class T> __inline__ __host__ __device__
T _Oppo(const T& a) { return -a; }

template<> __inline__ __host__ __device__
_SComplex _Oppo(const _SComplex& a)
{
    return make_cuComplex(-a.x, -a.y);
}

template<> __inline__ __host__ __device__
_DComplex _Oppo(const _DComplex& a)
{
    return make_cuDoubleComplex(-a.x, -a.y);
}

template<class T> __inline__ __host__ __device__
T _Inv(const T& a) { return static_cast<T>(1 / a); }

template<> __inline__ __host__ __device__
FLOAT _Inv(const FLOAT& a) { return 1.0f / a; }

template<> __inline__ __host__ __device__
DOUBLE _Inv(const DOUBLE& a) { return 1.0 / a; }

template<> __inline__ __host__ __device__
_SComplex _Inv(const _SComplex& a)
{
    return cuCdivf(make_cuComplex(1.0f, 0.0f), a);
}

template<> __inline__ __host__ __device__
_DComplex _Inv(const _DComplex& a)
{
    return cuCdiv(make_cuDoubleComplex(1.0, 0.0), a);
}

#pragma endregion

#pragma region Two element ops

//========== Add =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Add(const T1& a, const T2& b) { return a + static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Add(const _SComplex& a, const T2& b) { return make_cuComplex(a.x + static_cast<FLOAT>(b), a.y); }

template<class T2> __inline__ __host__ __device__
_DComplex _Add(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x + static_cast<DOUBLE>(b), a.y); }

template<class T1> __inline__ __host__ __device__
T1 _Add(const T1& a, const _SComplex& b) { return a + static_cast<T1>(cuCabsf(b)); }

template<class T1> __inline__ __host__ __device__
T1 _Add(const T1& a, const _DComplex& b) { return a + static_cast<T1>(cuCabs(b)); }

__inline__ __host__ __device__
_SComplex _Add(const _SComplex& a, const _SComplex& b) { return cuCaddf(a, b); }

__inline__ __host__ __device__
_DComplex _Add(const _DComplex& a, const _SComplex& b) { return make_cuDoubleComplex(a.x + static_cast<DOUBLE>(b.x), a.y + static_cast<DOUBLE>(b.y)); }

__inline__ __host__ __device__
_SComplex _Add(const _SComplex& a, const _DComplex& b) { return make_cuComplex(a.x + static_cast<FLOAT>(b.x), a.y + static_cast<FLOAT>(b.y)); }

__inline__ __host__ __device__
_DComplex _Add(const _DComplex& a, const _DComplex& b) { return cuCadd(a, b); }


//========== _Sub_L =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Sub(const T1& a, const T2& b) { return a - static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Sub(const _SComplex& a, const T2& b) { return make_cuComplex(a.x - static_cast<FLOAT>(b), a.y); }

template<class T2> __inline__ __host__ __device__
_DComplex _Sub(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x - static_cast<DOUBLE>(b), a.y); }

template<class T1> __inline__ __host__ __device__
T1 _Sub(const T1& a, const _SComplex b) { return a - static_cast<T1>(cuCabsf(b)); }

template<class T1> __inline__ __host__ __device__
T1 _Sub(const T1& a, const _DComplex& b) { return a - static_cast<T1>(cuCabs(b)); }

__inline__ __host__ __device__
_SComplex _Sub(const _SComplex& a, const _SComplex& b) { return cuCsubf(a, b); }

__inline__ __host__ __device__
_SComplex _Sub(const _SComplex& a, const _DComplex& b) { return make_cuComplex(a.x - static_cast<FLOAT>(b.x), a.y - static_cast<FLOAT>(b.y)); }

__inline__ __host__ __device__
_DComplex _Sub(const _DComplex& a, const _DComplex& b) { return cuCsub(a, b); }

__inline__ __host__ __device__
_DComplex _Sub(const _DComplex& a, const _SComplex& b) { return make_cuDoubleComplex(a.x - static_cast<DOUBLE>(b.x), a.y - static_cast<DOUBLE>(b.y)); }

//========== Mul =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Mul(const T1& a, const T2& b) { return a * static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Mul(const _SComplex& a, const T2& b) { return make_cuComplex(a.x * static_cast<FLOAT>(b), a.y * static_cast<FLOAT>(b)); }

template<class T2> __inline__ __host__ __device__
_DComplex _Mul(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x * static_cast<DOUBLE>(b), a.y * static_cast<DOUBLE>(b)); }

template<class T1> __inline__ __host__ __device__
T1 _Mul(const T1& a, const _SComplex b) { return a * static_cast<T1>(cuCabsf(b)); }

template<class T1> __inline__ __host__ __device__
T1 _Mul(const T1& a, const _DComplex& b) { return a * static_cast<T1>(cuCabs(b)); }

__inline__ __host__ __device__
_SComplex _Mul(const _SComplex& a, const _SComplex& b) { return cuCmulf(a, b); }

__inline__ __host__ __device__
_SComplex _Mul(const _SComplex& a, const _DComplex& b)
{
    return make_cuComplex(
        a.x * static_cast<FLOAT>(b.x) - a.y * static_cast<FLOAT>(b.y),
        a.x * static_cast<FLOAT>(b.y) + a.y * static_cast<FLOAT>(b.x)
        );
}

__inline__ __host__ __device__
_DComplex _Mul(const _DComplex& a, const _DComplex& b) { return cuCmul(a, b); }

__inline__ __host__ __device__
_DComplex _Mul(const _DComplex& a, const _SComplex& b)
{
    return make_cuDoubleComplex(
        a.x * static_cast<DOUBLE>(b.x) - a.y * static_cast<DOUBLE>(b.y),
        a.x * static_cast<DOUBLE>(b.y) + a.y * static_cast<DOUBLE>(b.x)
    );
}

//========== Div =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Div(const T1& a, const T2& b) { return a / static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Div(const _SComplex& a, const T2& b) { return make_cuComplex(a.x / static_cast<FLOAT>(b), a.y / static_cast<FLOAT>(b)); }

template<class T2> __inline__ __host__ __device__
_DComplex _Div(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x / static_cast<DOUBLE>(b), a.y / static_cast<DOUBLE>(b)); }

template<class T1> __inline__ __host__ __device__
T1 _Div(const T1& a, const _SComplex b) { return a / static_cast<T1>(cuCabsf(b)); }

template<class T1> __inline__ __host__ __device__
T1 _Div(const T1& a, const _DComplex& b) { return a / static_cast<T1>(cuCabs(b)); }

__inline__ __host__ __device__
_SComplex _Div(const _SComplex& a, const _SComplex& b) { return cuCdivf(a, b); }

__inline__ __host__ __device__
_SComplex _Div(const _SComplex& a, const _DComplex& b)
{
    FLOAT s = static_cast<FLOAT>(abs(b.x) + abs(b.y));
    FLOAT oos = 1.0f / s;
    const FLOAT ars = a.x * oos;
    const FLOAT ais = a.y * oos;
    const FLOAT brs = static_cast<FLOAT>(b.x) * oos;
    const FLOAT bis = static_cast<FLOAT>(b.y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    return make_cuComplex(
        ((ars * brs) + (ais * bis)) * oos,
        ((ais * brs) - (ars * bis)) * oos);
}

__inline__ __host__ __device__
_DComplex _Div(const _DComplex& a, const _DComplex& b) { return cuCdiv(a, b); }

__inline__ __host__ __device__
_DComplex _Div(const _DComplex& a, const _SComplex& b)
{
    DOUBLE s = abs(static_cast<DOUBLE>(b.x)) + abs(static_cast<DOUBLE>(b.y));
    DOUBLE oos = 1.0 / s;
    const DOUBLE ars = a.x * oos;
    const DOUBLE ais = a.y * oos;
    const DOUBLE brs = static_cast<DOUBLE>(b.x) * oos;
    const DOUBLE bis = static_cast<DOUBLE>(b.y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    return make_cuDoubleComplex(
        ((ars * brs) + (ais * bis)) * oos,
        ((ais * brs) - (ars * bis)) * oos);
}

//========== _Pow =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Pow(const T1& a, const T2& p) { return pow(a, static_cast<T1>(p)); }

template<class T2> __inline__ __host__ __device__
_SComplex _Pow(const _SComplex& a, const T2& p)
{
    const FLOAT fP = static_cast<FLOAT>(p);
    const FLOAT fArg = atan2(cuCimagf(a), cuCrealf(a)) * p;
    const FLOAT fAbs = pow(cuCabsf(a), fP);
    return make_cuComplex(cos(fArg) * fAbs, sin(fArg) * fAbs);
}

template<class T2> __inline__ __host__ __device__
_DComplex _Pow(const _DComplex& a, const T2& p)
{
    const DOUBLE fP = static_cast<DOUBLE>(p);
    const DOUBLE fArg = atan2(cuCimag(a), cuCreal(a)) * p;
    const DOUBLE fAbs = pow(cuCabs(a), fP);
    return make_cuDoubleComplex(cos(fArg) * fAbs, sin(fArg) * fAbs);
}

template<class T1> __inline__ __host__ __device__
T1 _Pow(const T1& a, const _SComplex b) { return pow(a, static_cast<T1>(cuCabsf(b))); }

template<class T1> __inline__ __host__ __device__
T1 _Pow(const T1& a, const _DComplex& b) { return pow(a, static_cast<T1>(cuCabs(b))); }

__inline__ __host__ __device__
_SComplex _Pow(const _SComplex& a, const _SComplex& b) 
{ 
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const FLOAT absa = cuCabsf(a);
    const FLOAT arga = atan2(cuCimagf(a), cuCrealf(a));
    const FLOAT factor = pow(absa, b.x) * exp(-arga * b.y);
    const FLOAT arg = b.y * log(absa) + arga * b.x;

    return make_cuComplex(cos(arg) * factor, sin(arg) * factor);
}

__inline__ __host__ __device__
_SComplex _Pow(const _SComplex& a, const _DComplex& b)
{
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const FLOAT absa = cuCabsf(a);
    const FLOAT arga = atan2(cuCimagf(a), cuCrealf(a));
    const FLOAT factor = pow(absa, static_cast<FLOAT>(b.x)) * exp(-arga * static_cast<FLOAT>(b.y));
    const FLOAT arg = static_cast<FLOAT>(b.y) * log(absa) + arga * static_cast<FLOAT>(b.x);

    return make_cuComplex(cos(arg) * factor, sin(arg) * factor);
}

__inline__ __host__ __device__
_DComplex _Pow(const _DComplex& a, const _DComplex& b) 
{ 
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const DOUBLE absa = cuCabs(a);
    const DOUBLE arga = atan2(cuCimag(a), cuCreal(a));
    const DOUBLE factor = pow(absa, b.x) * exp(-arga * b.y);
    const DOUBLE arg = b.y * log(absa) + arga * b.x;

    return make_cuDoubleComplex(cos(arg) * factor, sin(arg) * factor);
}

__inline__ __host__ __device__
_DComplex _Pow(const _DComplex& a, const _SComplex& b)
{
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const DOUBLE absa = cuCabs(a);
    const DOUBLE arga = atan2(cuCimag(a), cuCreal(a));
    const DOUBLE factor = pow(absa, static_cast<DOUBLE>(b.x)) * exp(-arga * static_cast<DOUBLE>(b.y));
    const DOUBLE arg = static_cast<DOUBLE>(b.y) * log(absa) + arga * static_cast<DOUBLE>(b.x);

    return make_cuDoubleComplex(cos(arg) * factor, sin(arg) * factor);
}

//========== _SubR =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _SubR(const T1& a, const T2& b) { return static_cast<T1>(b) - 1; }

template<class T2> __inline__ __host__ __device__
_SComplex _SubR(const _SComplex& a, const T2& b) { return make_cuComplex(static_cast<FLOAT>(b) - a.x, -a.y); }

template<class T2> __inline__ __host__ __device__
_DComplex _SubR(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(static_cast<DOUBLE>(b) - a.x, -a.y); }

template<class T1> __inline__ __host__ __device__
T1 _SubR(const T1& a, const _SComplex b) { return static_cast<T1>(cuCabsf(b)) - a; }

template<class T1> __inline__ __host__ __device__
T1 _SubR(const T1& a, const _DComplex& b) { return static_cast<T1>(cuCabs(b)) - a; }

__inline__ __host__ __device__
_SComplex _SubR(const _SComplex& a, const _SComplex& b) { return cuCsubf(b, a); }

__inline__ __host__ __device__
_SComplex _SubR(const _SComplex& a, const _DComplex& b) 
{ 
    return make_cuComplex(static_cast<FLOAT>(b.x) - a.x, static_cast<FLOAT>(b.y) - a.y);
}

__inline__ __host__ __device__
_DComplex _SubR(const _DComplex& a, const _DComplex& b) { return cuCsub(b, a); }

__inline__ __host__ __device__
_DComplex _SubR(const _DComplex& a, const _SComplex& b) { return make_cuDoubleComplex(b.x - static_cast<DOUBLE>(a.x), b.y - static_cast<DOUBLE>(a.y)); }


//========== DivR =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _DivR(const T1& a, const T2& b) { return static_cast<T1>(b) / a; }

template<class T2> __inline__ __host__ __device__
_SComplex _DivR(const _SComplex& a, const T2& b) 
{ 
    FLOAT s = abs(a.x) + abs(a.y);
    FLOAT oos = 1.0f / s;
    const FLOAT ars = static_cast<FLOAT>(b) * oos;
    const FLOAT brs = a.x * oos;
    const FLOAT bis = a.y * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    return make_cuComplex(
        (  (ars * brs)) * oos,
        (- (ars * bis)) * oos);
}

template<class T2> __inline__ __host__ __device__
_DComplex _DivR(const _DComplex& a, const T2& b) 
{ 
    DOUBLE s = abs(a.x) + abs(a.y);
    DOUBLE oos = 1.0 / s;
    const DOUBLE ars = static_cast<DOUBLE>(b) * oos;
    const DOUBLE brs = a.x * oos;
    const DOUBLE bis = a.y * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    return make_cuDoubleComplex(
        (   (ars * brs)) * oos,
        ( - (ars * bis)) * oos);
}

template<class T1> __inline__ __host__ __device__
T1 _DivR(const T1& a, const _SComplex b) { return static_cast<T1>(cuCabsf(b)) / a; }

template<class T1> __inline__ __host__ __device__
T1 _DivR(const T1& a, const _DComplex& b) { return static_cast<T1>(cuCabs(b)) / a; }

__inline__ __host__ __device__
_SComplex _DivR(const _SComplex& a, const _SComplex& b) { return cuCdivf(b, a); }

__inline__ __host__ __device__
_SComplex _DivR(const _SComplex& a, const _DComplex& b)
{
    FLOAT s = abs(a.x) + abs(a.y);
    FLOAT oos = 1.0f / s;
    const FLOAT ars = static_cast<FLOAT>(b.x) * oos;
    const FLOAT ais = static_cast<FLOAT>(b.y) * oos;
    const FLOAT brs = a.x * oos;
    const FLOAT bis = a.y * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    return make_cuComplex(
        ((ars * brs) + (ais * bis)) * oos,
        ((ais * brs) - (ars * bis)) * oos);
}

__inline__ __host__ __device__
_DComplex _DivR(const _DComplex& a, const _DComplex& b) { return cuCdiv(b, a); }

__inline__ __host__ __device__
_DComplex _DivR(const _DComplex& a, const _SComplex& b)
{
    DOUBLE s = abs(a.x) + abs(a.y);
    DOUBLE oos = 1.0 / s;
    const DOUBLE ars = static_cast<DOUBLE>(b.x) * oos;
    const DOUBLE ais = static_cast<DOUBLE>(b.y) * oos;
    const DOUBLE brs = a.x * oos;
    const DOUBLE bis = a.y * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    return make_cuDoubleComplex(
        ((ars * brs) + (ais * bis)) * oos,
        ((ais * brs) - (ars * bis)) * oos);
}

//========== _PowR =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _PowR(const T1& a, const T2& b) { return pow(static_cast<T1>(b), a); }

template<class T2> __inline__ __host__ __device__
_SComplex _PowR(const _SComplex& a, const T2& b)
{
    const FLOAT absa = abs(static_cast<FLOAT>(b));
    const FLOAT factor = pow(absa, a.x);
    const FLOAT arg = a.y * log(absa);

    return make_cuComplex(cos(arg) * factor, sin(arg) * factor);
}

template<class T2> __inline__ __host__ __device__
_DComplex _PowR(const _DComplex& a, const T2& b)
{
    const DOUBLE absa = abs(static_cast<DOUBLE>(b));
    const DOUBLE factor = pow(absa, a.x);
    const DOUBLE arg = a.y * log(absa);

    return make_cuDoubleComplex(cos(arg) * factor, sin(arg) * factor);
}

template<class T1> __inline__ __host__ __device__
T1 _PowR(const T1& a, const _SComplex b) 
{ 
    return static_cast<T1>(pow(cuCabsf(b), static_cast<FLOAT>(a)));
}

template<class T1> __inline__ __host__ __device__
T1 _PowR(const T1& a, const _DComplex& b) 
{ 
    return static_cast<T1>(pow(cuCabs(b), static_cast<DOUBLE>(a)));
}

__inline__ __host__ __device__
_SComplex _PowR(const _SComplex& a, const _SComplex& b)
{
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const FLOAT absa = cuCabsf(b);
    const FLOAT arga = atan2(cuCimagf(b), cuCrealf(b));
    const FLOAT factor = pow(absa, a.x) * exp(-arga * a.y);
    const FLOAT arg = a.y * log(absa) + arga * a.x;

    return make_cuComplex(cos(arg) * factor, sin(arg) * factor);
}

__inline__ __host__ __device__
_SComplex _PowR(const _SComplex& a, const _DComplex& b)
{
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const FLOAT absa = static_cast<FLOAT>(cuCabs(b));
    const FLOAT arga = static_cast<FLOAT>(atan2(cuCimag(b), cuCreal(b)));
    const FLOAT factor = pow(absa, a.x) * exp(-arga * a.y);
    const FLOAT arg = a.y * log(absa) + arga * a.x;

    return make_cuComplex(cos(arg) * factor, sin(arg) * factor);
}

__inline__ __host__ __device__
_DComplex _PowR(const _DComplex& a, const _DComplex& b)
{
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const DOUBLE absa = cuCabs(b);
    const DOUBLE arga = atan2(cuCimag(b), cuCreal(b));
    const DOUBLE factor = pow(absa, a.x) * exp(-arga * a.y);
    const DOUBLE arg = a.y * log(absa) + arga * a.x;

    return make_cuDoubleComplex(cos(arg) * factor, sin(arg) * factor);
}

__inline__ __host__ __device__
_DComplex _PowR(const _DComplex& a, const _SComplex& b)
{
    //Abs[a]^Re[b] Exp[-Arg[a] Im[b]](Cos[Im[b] Log[Abs[a]] + Arg[a] Re[b]] + I Sin[Im[b] Log[Abs[a]] + Arg[a] Re[b]])
    const DOUBLE rb = static_cast<DOUBLE>(b.x);
    const DOUBLE ib = static_cast<DOUBLE>(b.y);
    const DOUBLE absa = sqrt(rb * rb + ib * ib);
    const DOUBLE arga = atan2(static_cast<DOUBLE>(cuCimagf(b)), static_cast<DOUBLE>(cuCrealf(b)));
    const DOUBLE factor = pow(absa, a.x) * exp(-arga * a.y);
    const DOUBLE arg = a.y * log(absa) + arga * a.x;

    return make_cuDoubleComplex(cos(arg) * factor, sin(arg) * factor);
}


#pragma endregion

#pragma region Logs

template<class T> inline void LogValue(const T& )
{
    appGeneral(_T("Not supported"));
}

template<> inline void LogValue(const FLOAT& v)
{
    appGeneral(_T("%2.12f"), v);
}

template<> inline void LogValue(const DOUBLE& v)
{
    appGeneral(_T("%2.12f"), v);
}

template<> inline void LogValue(const INT& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const UINT& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const BYTE& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const SBYTE& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const USHORT& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const SHORT& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const QWORD& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const SQWORD& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const _SComplex& v)
{
    appGeneral(_T("%2.12f %s %2.12f I"), 
        v.x,
        v.y >= 0.0f ? _T("+") : _T("-"),
        _Abs(v.y));
}

template<> inline void LogValue(const _DComplex& v)
{
    appGeneral(_T("%2.12f %s %2.12f I"),
        v.x,
        v.y >= 0.0 ? _T("+") : _T("-"),
        _Abs(v.y));
}

#pragma endregion


__END_NAMESPACE

#endif //#ifndef _CUDACOMPLEXFUNCTION_H_

//=============================================================================
// END OF FILE
//=============================================================================
