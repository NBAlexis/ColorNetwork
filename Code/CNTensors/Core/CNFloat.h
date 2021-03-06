//=============================================================================
// FILENAME : CLGFloat.h
// 
// DESCRIPTION:
// Add precsion for floats
//
// REVISION:
//  [31/05/2020 nbale]
//=============================================================================
#ifndef _CNFLOAT_H_
#define _CNFLOAT_H_

/*
#if _CN_DOUBLEFLOAT

#define _sqrt sqrt
#define _log log
#define _exp exp
#define _pow pow
#define _sin sin
#define _cos cos
#define __div(a, b) ((a) / (b))
#define __rcp(a) (F(1.0) / (a))
#define _hostlog log
#define _hostlog10 log10
#define _hostexp exp
#define _hostsqrt sqrt

#define _atan2 atan2
#define _make_cuComplex make_cuDoubleComplex
#define _cuCaddf cuCadd
#define _cuCmulf cuCmul
#define _cuCsubf cuCsub
#define _cuConjf cuConj
#define _cuCrealf cuCreal
#define _cuCimagf cuCimag
#define _cuCabsf cuCabs
#define _cuCdivf cuCdiv
#define F(v) v
#if defined(__cplusplus) && defined(__CUDACC__)
#define _floor2int __double2int_rd
#define _round2int __double2int_rn
#else
#define _floor2int(a) static_cast<INT>(floor(a))
#define _round2int(a) static_cast<INT>(round(a))
#endif

#else

#if defined(__cplusplus) && defined(__CUDACC__)
#define _sqrt __fsqrt_rn
#define _log __logf
#define _exp __expf
#define _pow __powf
#define _sin __sinf
#define _cos __cosf
#define __div __fdividef
#define __rcp __frcp_rn
#else
//the __function is Intrinsic Functions which can be only used in device
#define _sqrt sqrtf
#define _log logf
#define _exp expf
#define _pow powf
#define _sin sinf
#define _cos cosf
#define __div(a, b) ((a) / (b))
#define __rcp(a) (F(1.0) / (a))
#endif

#define _hostlog logf
#define _hostlog10 log10f
#define _hostexp expf
#define _hostsqrt sqrtf

#define _atan2 atan2f
#define _make_cuComplex make_cuComplex
#define _cuCaddf cuCaddf
#define _cuCmulf cuCmulf
#define _cuCsubf cuCsubf
#define _cuConjf cuConjf
#define _cuCrealf cuCrealf
#define _cuCimagf cuCimagf
#define _cuCabsf cuCabsf
#define _cuCdivf cuCdivf
#define F(v) v##f
#if defined(__cplusplus) && defined(__CUDACC__)
#define _floor2int __float2int_rd
#define _round2int __float2int_rn
#else
#define _floor2int(a) static_cast<INT>(floor(a))
#define _round2int(a) static_cast<INT>(round(a))
#endif

#endif
*/

#if _CN_DOUBLEFLOAT

#define _CN_FLT_MIN_ 1E-50   //When smaller than this, sqrt, divide will become nan

#define _CN_FLT_DECIMAL_DIG  17                      // # of decimal digits of rounding precision
#define _CN_FLT_DIG          15                      // # of decimal digits of precision
#define _CN_FLT_EPSILON      2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0
#define _CN_FLT_HAS_SUBNORM  1                       // type does support subnormal numbers
#define _CN_FLT_MANT_DIG     53                      // # of bits in mantissa
#define _CN_FLT_MAX          1.7976931348623158e+308 // max value
#define _CN_FLT_MAX_10_EXP   308                     // max decimal exponent
#define _CN_FLT_MAX_EXP      1024                    // max binary exponent
#define _CN_FLT_MIN          2.2250738585072014e-308 // min positive value
#define _CN_FLT_MIN_10_EXP   (-307)                  // min decimal exponent
#define _CN_FLT_MIN_EXP      (-1021)                 // min binary exponent
#define _CN_FLT_RADIX        2                       // exponent radix
#define _CN_FLT_TRUE_MIN     4.9406564584124654e-324 // min positive value

#else

//They are not defined in GCC, so we define them explicitly
#define _CN_FLT_MIN_ 1E-22F   //When smaller than this, sqrt, divide will become nan

#define _CN_FLT_DECIMAL_DIG  9                       // # of decimal digits of rounding precision
#define _CN_FLT_DIG          6                       // # of decimal digits of precision
#define _CN_FLT_EPSILON      1.192092896e-07F        // smallest such that 1.0+FLT_EPSILON != 1.0
#define _CN_FLT_HAS_SUBNORM  1                       // type does support subnormal numbers
#define _CN_FLT_GUARD        0
#define _CN_FLT_MANT_DIG     24                      // # of bits in mantissa
#define _CN_FLT_MAX          3.402823466e+38F        // max value
#define _CN_FLT_MAX_10_EXP   38                      // max decimal exponent
#define _CN_FLT_MAX_EXP      128                     // max binary exponent
#define _CN_FLT_MIN          1.175494351e-38F        // min normalized positive value
#define _CN_FLT_MIN_10_EXP   (-37)                   // min decimal exponent
#define _CN_FLT_MIN_EXP      (-125)                  // min binary exponent
#define _CN_FLT_NORMALIZE    0
#define _CN_FLT_RADIX        2                       // exponent radix
#define _CN_FLT_TRUE_MIN     1.401298464e-45F        // min positive value

#endif

#pragma region Constants

#define _zerocs (make_cuComplex(0.0f, 0.0f))
#define _onecs (make_cuComplex(1.0f, 0.0f))
#define _imgcs (make_cuComplex(0.0f, 1.0f))

#define _zerocd (make_cuDoubleComplex(0.0f, 0.0f))
#define _onecd (make_cuDoubleComplex(1.0f, 0.0f))
#define _imgcd (make_cuDoubleComplex(0.0f, 1.0f))

//Those are constants we are using

//save some constant memory of cuda?
#define PI (3.141592653589)
#define PISQ (9.8696044010893586188344909998761511353137)

// = 1/4294967296UL
#define AM (0.00000000023283064365386963)
// = _sqrt(2)
#define SQRT2 (1.4142135623730951)
// = 1 / _sqrt(2), or _sqrt(2)/2
#define InvSqrt2 (0.7071067811865475)
// = 2.0f * PI
#define PI2 (6.283185307179586)

// 1.0f / _sqrt(3)
#define InvSqrt3 (0.5773502691896258)
// 2.0f / _sqrt(3)
#define InvSqrt3_2 (1.1547005383792517)

#define OneOver6 (0.16666666666666666666666666666667)
#define OneOver24 (0.04166666666666666666666666666667)

//typically, 0.3-0.5 - arXiv:002.4232
#define OmelyanLambda2 (0.38636665500756728)

#pragma endregion

__BEGIN_NAMESPACE

typedef cuDoubleComplex _DComplex;
typedef cuComplex _SComplex;

//NOTE, _Complex is already a keyword in GCC
/*
#if _CN_DOUBLEFLOAT

typedef double Real;
typedef cuDoubleComplex CNComplex;

#else

typedef float Real;
typedef cuComplex CNComplex;

#endif
*/

__END_NAMESPACE

#endif//#ifndef _CNFLOAT_H_

//=============================================================================
// END OF FILE
//=============================================================================