//=============================================================================
// FILENAME : CudaComplexFunction.h
// 
// DESCRIPTION:
// Add some function to CNComplex where it does not have yet
//
//
// REVISION:
//  [12/6/2018 nbale]
//=============================================================================

#ifndef _CUDACOMPLEXFUNCTION_H_
#define _CUDACOMPLEXFUNCTION_H_

__BEGIN_NAMESPACE

#pragma region One element ops

template<class T> __inline__ __host__ __device__
Real _Abs(const T& a) { return abs(static_cast<Real>(a)); }

template<> __inline__ __host__ __device__
Real _Abs(const CNComplex& a)
{
    return _cuCabsf(a);
}

template<class T> __inline__ __host__ __device__
Real _Re(const T& a) { return static_cast<Real>(a); }

template<> __inline__ __host__ __device__
Real _Re(const CNComplex& a)
{
    return a.x;
}

template<class T> __inline__ __host__ __device__
constexpr Real _Im(const T& a) { return F(0.0); }

template<> __inline__ __host__ __device__
Real _Im(const CNComplex& a)
{
    return a.y;
}

template<class T> __inline__ __host__ __device__
Real _AbsSq(const T& a) { return static_cast<Real>(a * a); }

template<> __inline__ __host__ __device__
Real _AbsSq(const CNComplex& a)
{
    return a.x * a.x + a.y * a.y;
}

template<class T> __inline__ __host__ __device__
T _Conj(const T& a) { return a; }

template<> __inline__ __host__ __device__
CNComplex _Conj(const CNComplex& a)
{
    return _cuConjf(a);
}

template<class T> __inline__ __host__ __device__
constexpr Real _Arg(const T& a) { return F(0.0); }

template<> __inline__ __host__ __device__
Real _Arg(const CNComplex& a)
{
    return _atan2(_cuCimagf(a), _cuCrealf(a));
}

template<class T> __inline__ __host__ __device__
T _Pow(const T& a, Real p) { return _pow(a, p); }

template<> __inline__ __host__ __device__
CNComplex _Pow(const CNComplex& a, Real p)
{
    const Real fArg = _Arg(a) * p;
    const Real fAbs = _pow(_cuCabsf(a), p);
    return _make_cuComplex(_cos(fArg) * fAbs, _sin(fArg) * fAbs);
}

template<class T> __inline__ __host__ __device__
T _Exp(const T& a) { return _exp(a); }

template<> __inline__ __host__ __device__
CNComplex _Exp(const CNComplex& a)
{
    const Real factor = _exp(a.x);
    return _make_cuComplex(factor * _cos(a.y), factor * _sin(a.y));
}

template<class T> __inline__ __host__ __device__
T _Log(const T& a) { return _log(a); }

template<class T> __inline__ __host__ __device__
CNComplex _Log(const CNComplex& a)
{
    const Real fArg = _Arg(a);
    return _make_cuComplex(_log(_cuCabsf(a)), fArg > PI ? fArg - PI2 : fArg);
}

template<class T> __inline__ __host__ __device__
T _Sqrt(const T& a) { return _sqrt(a); }

template<class T> __inline__ __host__ __device__
CNComplex _Sqrt(const CNComplex& a)
{
    const Real fRadius = _cuCabsf(a);
    const Real fCosA = __div(a.x, fRadius);
    CNComplex out;
    out.x = _sqrt(F(0.5) * fRadius * (fCosA + F(1.0)));
    out.y = _sqrt(F(0.5) * fRadius * (F(1.0) - fCosA));
    if (a.y < F(0.0))
        out.y *= -F(1.0);

    return out;
}

template<class T> __inline__ __host__ __device__
T _Sin(const T& a) { return _sin(a); }

template<class T> __inline__ __host__ __device__
CNComplex _Sin(const CNComplex& a)
{
    const CNComplex numerator = _cuCsubf(
        _Exp(_make_cuComplex(-a.y, a.x)), 
        _Exp(_make_cuComplex(a.y, -a.x)));

    return _make_cuComplex(numerator.y * F(0.5), numerator.x * F(-0.5));
}

template<class T> __inline__ __host__ __device__
T _Cos(const T& a) { return _cos(a); }

template<class T> __inline__ __host__ __device__
CNComplex _Cos(const CNComplex& a)
{
    const CNComplex numerator = _cuCaddf(
        _Exp(_make_cuComplex(-a.y, a.x)),
        _Exp(_make_cuComplex(a.y, -a.x)));

    return _make_cuComplex(numerator.x * F(0.5), numerator.y * F(0.5));
}

#pragma endregion

#pragma region Two element ops

template<class T1, class T2> __inline__ __host__ __device__
T1 _Add(const T1& a, const T2& b) { return a + static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
CNComplex _Add(const CNComplex& a, const T2& b) { return _make_cuComplex(b + a.x, a.y); }

template<> __inline__ __host__ __device__
CNComplex _Add(const CNComplex& a, const CNComplex& b) { return _cuCaddf(a, b); }

template<class T1, class T2> __inline__ __host__ __device__
T1 _Sub_L(const T1& a, const T2& b) { return a - static_cast<T1>(b); }

template<class T> __inline__ __host__ __device__
CNComplex _Sub_L(const CNComplex& a, const T& b) { return _make_cuComplex(a.x - b, a.y); }

template<> __inline__ __host__ __device__
CNComplex _Sub_L(const CNComplex& a, const CNComplex& b) { return _cuCsubf(a, b); }

template<class T1, class T2> __inline__ __host__ __device__
T2 _Sub_R(const T1& a, const T2& b) { return a - static_cast<T1>(b); }

template<class T> __inline__ __host__ __device__
CNComplex _Sub_R(const T& a, const CNComplex& b) { return _make_cuComplex(a - b.x, -b.y); }

template<> __inline__ __host__ __device__
CNComplex _Sub_R(const CNComplex& a, const CNComplex& b) { return _Sub_L(a, b); }

template<class T1, class T2> __inline__ __host__ __device__
T1 _Mul(const T1& a, const T2& b) { return a * static_cast<T1>(b); }

template<class T> __inline__ __host__ __device__
CNComplex _Mul(const CNComplex& a, const T& b) { return _make_cuComplex(a.x * b, a.y * b); }

template<> __inline__ __host__ __device__
CNComplex _Mul(const CNComplex& a, const CNComplex& b) { return _cuCmulf(a, b); }

template<class T1, class T2> __inline__ __host__ __device__
T1 _Div_L(const T1& a, const T2& b) { return a / static_cast<T1>(b); }

template<class T> __inline__ __host__ __device__
CNComplex _Div_L(const CNComplex& a, const T& b) { return _make_cuComplex(a.x / b, a.y / b); }

template<> __inline__ __host__ __device__
CNComplex _Div_L(const CNComplex& a, const CNComplex& b) { return _cuCdivf(a, b); }

template<class T1, class T2> __inline__ __host__ __device__
T2 _Div_R(const T1& a, const T2& b) { return a / static_cast<T1>(b); }

template<class T> __inline__ __host__ __device__
CNComplex _Div_R(const T& a, const CNComplex& b) { return _Div_L(_make_cuComplex(a, F(0.0)), b); }

template<> __inline__ __host__ __device__
CNComplex _Div_R(const CNComplex& a, const CNComplex& b) { return _Div_L(a, b); }

#pragma endregion

#pragma region Logs

template<class T> inline void LogValue(const T& )
{
    appGeneral(_T("Not supported"));
}

template<> inline void LogValue(const Real& v)
{
    appGeneral(_T("%2.20f"), v);
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

template<> inline void LogValue(const WORD& v)
{
    appGeneral(_T("%d"), v);
}

template<> inline void LogValue(const SWORD& v)
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

template<> inline void LogValue(const CNComplex& v)
{
    appGeneral(_T("%2.20f %s %2.20f I"), 
        v.x,
        v.y >= F(0.0) ? _T("+") : _T("-"),
        _Abs(v.y));
}

#pragma endregion


__END_NAMESPACE

#endif //#ifndef _CUDACOMPLEXFUNCTION_H_

//=============================================================================
// END OF FILE
//=============================================================================
