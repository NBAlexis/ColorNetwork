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

__BEGIN_NAMESPACE

#pragma region One element ops

template<class T> __inline__ __host__ __device__
void _Zero(T& a) { a = static_cast<T>(0); }

template<> __inline__ __host__ __device__
void _Zero(_SComplex& a) { a = _zerocs; }

template<> __inline__ __host__ __device__
void _Zero(_DComplex& a) { a = _zerocd; }

template<class T> __inline__ __host__ __device__
void _One(T& a) { a = static_cast<T>(1); }

template<> __inline__ __host__ __device__
void _One(_SComplex& a) { a = _onecs; }

template<> __inline__ __host__ __device__
void _One(_DComplex& a) { a = _onecd; }

template<class T> __inline__ __host__ __device__
T _Abs(const T& a) { return abs(a); }

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
FLOAT _AbsSq(const _SComplex& a) { return a.x * a.x + a.y * a.y; }

__inline__ __host__ __device__
DOUBLE _AbsSq(const _DComplex& a) { return a.x * a.x + a.y * a.y; }

template<class T> __inline__ __host__ __device__
T _Conj(const T& a) { return a; }

template<> __inline__ __host__ __device__
_SComplex _Conj(const _SComplex& a) { return cuConjf(a); }

template<> __inline__ __host__ __device__
_DComplex _Conj(const _DComplex& a) { return cuConj(a); }

template<class T> __inline__ __host__ __device__
constexpr T _Arg(const T& a) { return static_cast<T>(0); }

__inline__ __host__ __device__
FLOAT _Arg(const _SComplex& a) { return atan2(cuCimagf(a), cuCrealf(a)); }

__inline__ __host__ __device__
DOUBLE _Arg(const _DComplex& a) { return atan2(cuCimag(a), cuCreal(a)); }

template<class T1, class T2> __inline__ __host__ __device__
T1 _Pow(const T1& a, T2 p) { return pow(a, p); }

template<class T2> __inline__ __host__ __device__
_SComplex _Pow(const _SComplex& a, T2 p)
{
    const FLOAT fP = static_cast<FLOAT>(p);
    const FLOAT fArg = _Arg(a) * p;
    const FLOAT fAbs = pow(cuCabsf(a), fP);
    return make_cuComplex(cos(fArg) * fAbs, sin(fArg) * fAbs);
}

template<class T2> __inline__ __host__ __device__
_DComplex _Pow(const _DComplex& a, T2 p)
{
    const DOUBLE fP = static_cast<DOUBLE>(p);
    const DOUBLE fArg = _Arg(a) * p;
    const DOUBLE fAbs = pow(cuCabs(a), fP);
    return make_cuDoubleComplex(cos(fArg) * fAbs, sin(fArg) * fAbs);
}

template<class T> __inline__ __host__ __device__
T _Exp(const T& a) { return exp(a); }

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
T _Log(const T& a) { return _log(a); }

template<class T> __inline__ __host__ __device__
_SComplex _Log(const _SComplex& a)
{
    const FLOAT fArg = _Arg(a);
    return make_cuComplex(log(cuCabsf(a)), fArg > PIF ? fArg - PI2F : fArg);
}

template<class T> __inline__ __host__ __device__
_DComplex _Log(const _DComplex& a)
{
    const DOUBLE fArg = _Arg(a);
    return make_cuDoubleComplex(log(cuCabs(a)), fArg > PI ? fArg - PI2 : fArg);
}

template<class T> __inline__ __host__ __device__
T _Sqrt(const T& a) { return sqrt(a); }

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
T _Sin(const T& a) { return static_cast<T>(sin(a)); }

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
T _Cos(const T& a) { return cos(a); }

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

#pragma endregion

#pragma region Two element ops

//========== Add =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Add(const T1& a, const T2& b) { return a + static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Add(const _SComplex& a, const T2& b) { return make_cuComplex(a.x + static_cast<FLOAT>(b), a.y); }

template<class T2> __inline__ __host__ __device__
_DComplex _Add(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x + static_cast<DOUBLE>(b), a.y); }

template<> __inline__ __host__ __device__
_SComplex _Add(const _SComplex& a, const _SComplex& b) { return cuCaddf(a, b); }

template<> __inline__ __host__ __device__
_SComplex _Add(const _SComplex& a, const _DComplex& b) { return make_cuComplex(a.x + static_cast<FLOAT>(b.x), a.y + static_cast<FLOAT>(b.y)); }

template<> __inline__ __host__ __device__
_DComplex _Add(const _DComplex& a, const _DComplex& b) { return cuCadd(a, b); }

template<> __inline__ __host__ __device__
_DComplex _Add(const _DComplex& a, const _SComplex& b) { return make_cuDoubleComplex(a.x + static_cast<DOUBLE>(b.x), a.y + static_cast<DOUBLE>(b.y)); }

//========== _Sub_L =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Sub(const T1& a, const T2& b) { return a - static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Sub(const _SComplex& a, const T2& b) { return make_cuComplex(a.x - static_cast<FLOAT>(b), a.y); }

template<class T2> __inline__ __host__ __device__
_DComplex _Sub(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x - static_cast<DOUBLE>(b), a.y); }

template<> __inline__ __host__ __device__
_SComplex _Sub(const _SComplex& a, const _SComplex& b) { return cuCsubf(a, b); }

template<> __inline__ __host__ __device__
_SComplex _Sub(const _SComplex& a, const _DComplex& b) { return make_cuComplex(a.x - static_cast<FLOAT>(b.x), a.y - static_cast<FLOAT>(b.y)); }

template<> __inline__ __host__ __device__
_DComplex _Sub(const _DComplex& a, const _DComplex& b) { return cuCsub(a, b); }

template<> __inline__ __host__ __device__
_DComplex _Sub(const _DComplex& a, const _SComplex& b) { return make_cuDoubleComplex(a.x - static_cast<DOUBLE>(b.x), a.y - static_cast<DOUBLE>(b.y)); }

//========== _Sub_R =============
template<class T1, class T2> __inline__ __host__ __device__
T2 _Sub_R(const T1& a, const T2& b) { return static_cast<T2>(a) - b; }

template<class T1> __inline__ __host__ __device__
_SComplex _Sub_R(const T1& a, const _SComplex& b) { return make_cuComplex(static_cast<FLOAT>(a) - b.x, -b.y); }

template<class T1> __inline__ __host__ __device__
_DComplex _Sub_R(const T1& a, const _DComplex& b) { return make_cuDoubleComplex(static_cast<DOUBLE>(a) - b.x, -b.y); }

template<> __inline__ __host__ __device__
_SComplex _Sub_R(const _SComplex& a, const _SComplex& b) { return cuCsubf(a, b); }

template<> __inline__ __host__ __device__
_DComplex _Sub_R(const _SComplex& a, const _DComplex& b) { return make_cuDoubleComplex(static_cast<DOUBLE>(a.x) - b.x, static_cast<DOUBLE>(a.y) -b.y); }

template<> __inline__ __host__ __device__
_DComplex _Sub_R(const _DComplex& a, const _DComplex& b) { return cuCsub(a, b); }

template<> __inline__ __host__ __device__
_SComplex _Sub_R(const _DComplex& a, const _SComplex& b) { return make_cuComplex(static_cast<FLOAT>(a.x) - b.x, static_cast<FLOAT>(a.y) - b.y); }

//========== Mul =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Mul(const T1& a, const T2& b) { return a * static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Mul(const _SComplex& a, const T2& b) { return make_cuComplex(a.x * static_cast<FLOAT>(b), a.y * static_cast<FLOAT>(b)); }

template<class T2> __inline__ __host__ __device__
_DComplex _Mul(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x * static_cast<DOUBLE>(b), a.y * static_cast<DOUBLE>(b)); }

template<> __inline__ __host__ __device__
_SComplex _Mul(const _SComplex& a, const _SComplex& b) { return cuCmulf(a, b); }

template<> __inline__ __host__ __device__
_SComplex _Mul(const _SComplex& a, const _DComplex& b)
{
    return make_cuComplex(
        a.x * static_cast<FLOAT>(b.x) - a.y * static_cast<FLOAT>(b.y),
        a.x * static_cast<FLOAT>(b.y) + a.y * static_cast<FLOAT>(b.x)
        );
}

template<> __inline__ __host__ __device__
_DComplex _Mul(const _DComplex& a, const _DComplex& b) { return cuCmul(a, b); }

template<> __inline__ __host__ __device__
_DComplex _Mul(const _DComplex& a, const _SComplex& b)
{
    return make_cuDoubleComplex(
        a.x * static_cast<DOUBLE>(b.x) - a.y * static_cast<DOUBLE>(b.y),
        a.x * static_cast<DOUBLE>(b.y) + a.y * static_cast<DOUBLE>(b.x)
    );
}

//========== Div_L =============
template<class T1, class T2> __inline__ __host__ __device__
T1 _Div(const T1& a, const T2& b) { return a / static_cast<T1>(b); }

template<class T2> __inline__ __host__ __device__
_SComplex _Div(const _SComplex& a, const T2& b) { return make_cuComplex(a.x / static_cast<FLOAT>(b), a.y / static_cast<FLOAT>(b)); }

template<class T2> __inline__ __host__ __device__
_DComplex _Div(const _DComplex& a, const T2& b) { return make_cuDoubleComplex(a.x / static_cast<DOUBLE>(b), a.y / static_cast<DOUBLE>(b)); }

template<> __inline__ __host__ __device__
_SComplex _Div(const _SComplex& a, const _SComplex& b) { return cuCdivf(a, b); }

template<> __inline__ __host__ __device__
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

template<> __inline__ __host__ __device__
_DComplex _Div(const _DComplex& a, const _DComplex& b) { return cuCdiv(a, b); }

template<> __inline__ __host__ __device__
_DComplex _Div(const _DComplex& a, const _SComplex& b)
{
    DOUBLE s = abs(b.x) + abs(b.y);
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

//========== Div_R =============
template<class T1, class T2> __inline__ __host__ __device__
T2 _Div_R(const T1& a, const T2& b) { return static_cast<T2>(a) / b; }

template<class T1> __inline__ __host__ __device__
_SComplex _Div_R(const T1& a, const _SComplex& b)
{
    FLOAT s = abs(b.x) + abs(b.y);
    FLOAT oos = 1.0f / s;
    const FLOAT ars = a * oos;
    const FLOAT brs = static_cast<FLOAT>(b.x) * oos;
    const FLOAT bis = static_cast<FLOAT>(b.y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    return make_cuComplex(ars * brs * oos, - ars * bis * oos);
}

template<class T2> __inline__ __host__ __device__
_DComplex _Div_R(const T2& a, const _DComplex& b)
{
    DOUBLE s = abs(b.x) + abs(b.y);
    DOUBLE oos = 1.0 / s;
    const DOUBLE ars = a * oos;
    const DOUBLE brs = static_cast<DOUBLE>(b.x) * oos;
    const DOUBLE bis = static_cast<DOUBLE>(b.y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    return make_cuDoubleComplex(ars * brs * oos, - ars * bis * oos);
}

template<> __inline__ __host__ __device__
_SComplex _Div_R(const _SComplex& a, const _SComplex& b) { return cuCdivf(a, b); }

template<> __inline__ __host__ __device__
_DComplex _Div_R(const _SComplex& a, const _DComplex& b)
{
    DOUBLE s = abs(b.x) + abs(b.y);
    DOUBLE oos = 1.0 / s;
    const DOUBLE ars = static_cast<DOUBLE>(a.x) * oos;
    const DOUBLE ais = static_cast<DOUBLE>(a.y) * oos;
    const DOUBLE brs = b.x * oos;
    const DOUBLE bis = b.y * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    return make_cuDoubleComplex(
        ((ars * brs) + (ais * bis)) * oos,
        ((ais * brs) - (ars * bis)) * oos);
}

template<> __inline__ __host__ __device__
_DComplex _Div_R(const _DComplex& a, const _DComplex& b) { return cuCdiv(a, b); }

template<> __inline__ __host__ __device__
_SComplex _Div_R(const _DComplex& a, const _SComplex& b)
{
    FLOAT s = abs(b.x) + abs(b.y);
    FLOAT oos = 1.0f / s;
    const FLOAT ars = static_cast<FLOAT>(a.x) * oos;
    const FLOAT ais = static_cast<FLOAT>(a.y) * oos;
    const FLOAT brs = b.x * oos;
    const FLOAT bis = b.y * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    return make_cuComplex(
        ((ars * brs) + (ais * bis)) * oos,
        ((ais * brs) - (ars * bis)) * oos);
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
