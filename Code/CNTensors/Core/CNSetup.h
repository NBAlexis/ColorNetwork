//=============================================================================
// FILENAME : CNSetup.h
// 
// DESCRIPTION:
// This is the file for building options
//
// REVISION:
//  [24/04/2020 nbalexis]
//=============================================================================
#ifndef _CNSETUP_H_
#define _CNSETUP_H_

//No support for unicode, the unicode is not well supported in CUDA
//#define _CN_UNICODE 1

#ifdef DEBUG
#define _CN_DEBUG 1
#endif

//Note: Important!
//This is the tag for windows, msvc specific
//Ignore _MSC_VER, which is just for Visual Studio IDE specific, and should be harmless
#ifdef WIN64
#define _CN_WIN 1
#elif defined(_WIN64)
#define _CN_WIN 1
#endif

//_CLG_USE_LAUNCH_BOUND = 0 or 1.
//NOTE: If the regcount required is out-numbered, sometimes, there is NO error message!
//So, either be sure to build with _CN_USE_LAUNCH_BOUND = 1, or reduce the thread count
//reduce the thread count is expansive, so _CLG_USE_LAUNCH_BOUND = 1 is recommanded
//It's better to complie using the maximum thread per block of the device of the computer.
#if _CN_DEBUG
#define _CN_USE_LAUNCH_BOUND 0
#else
#define _CN_USE_LAUNCH_BOUND 1
#endif

#define _CN_GLOBALE_AS_CONST 0

#if _CN_USE_LAUNCH_BOUND
constexpr unsigned int MAX_THREAD = 1024;
constexpr unsigned int BOUND_THREAD = 1024;
constexpr unsigned int BOUND_BLOCK = 1;
#else
constexpr unsigned int MAX_THREAD = 1024;
constexpr unsigned int BOUND_THREAD = 1024;
constexpr unsigned int BOUND_BLOCK = 1;
#endif


#endif //#ifndef _CNSETUP_H_

//=============================================================================
// END OF FILE
//=============================================================================