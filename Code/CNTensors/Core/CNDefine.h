//=============================================================================
// FILENAME : CLGDefine.h
// 
// DESCRIPTION:
// This is the file for some common definations
//
// REVISION:
//  [24/04/2020 nbalexis]
//=============================================================================
#ifndef _CNDEFINE_H_
#define _CNDEFINE_H_

#pragma region Namespace

#ifdef  __NAMESPACE
#undef  __NAMESPACE
#endif
#define __NAMESPACE                ColorNetwork

#ifdef  __GVERSION
#undef  __GVERSION
#endif
#define __GVERSION      (1)

#ifdef  __GVERSION_S
#undef  __GVERSION_S
#endif
#define __GVERSION_S    (0)

#ifdef  __BEGIN_NAMESPACE
#undef  __BEGIN_NAMESPACE
#endif
#define __BEGIN_NAMESPACE        namespace __NAMESPACE{

#ifdef  __END_NAMESPACE
#undef  __END_NAMESPACE
#endif
#define __END_NAMESPACE            }

#ifdef  __USE_NAMESPACE
#undef  __USE_NAMESPACE
#endif
#define __USE_NAMESPACE            using namespace __NAMESPACE;

#pragma endregion Namespace


#pragma region Function call

#if _CN_WIN
# define __DLL_IMPORT            __declspec(dllimport)
# define CNAPIPRIVATE
# define __DLL_EXPORT            __declspec(dllexport)
# define __IMPORT_LIB(libname)    comment(lib, libname)
# undef FORCEINLINE
# undef CDECL
# define FORCEINLINE             __forceinline
# define CDECL                   __cdecl

# define SUPPORTS_PRAGMA_PACK 1
# define __PACK_PUSH                pack(push, 8)
# define __PACK_POP                pack(pop)
#else
# define __DLL_IMPORT            
# define CNAPIPRIVATE
# define __DLL_EXPORT            
# define __IMPORT_LIB(libname)    
# undef FORCEINLINE
# undef CDECL
# define FORCEINLINE             inline
# define CDECL                   __cdecl

# define SUPPORTS_PRAGMA_PACK 0
# define __PACK_PUSH            
# define __PACK_POP                
#endif

#pragma endregion

#pragma region Helpers

#define ARRAY_COUNT( aarray ) \
    ( sizeof(aarray) / sizeof((aarray)[0]) )

#define appSafeFree(p)        {if(p){free(p); p=NULL;}}
#define appSafeDelete(p)        {if(p){delete p; p=NULL;}}
#define appSafeDeleteArray(p)    {if(p){delete[] p; p=NULL;}}

#define UN_USE(a) (void)a

//aligned alloca
//extern "C" void* __cdecl _alloca(size_t);
#define appAlloca(size) ((0 == size) ? 0 : alloca((size+7)&~7))

#define __CN_FORCEOBJ_HEAD(name) \
struct CNAPI name##helper \
{ \
    name##helper(); \
}; \
static name##helper s_##name##helper;

#define __CN_FORCEOBJ_CPP(name) \
    name##helper::name##helper() {} 

#pragma endregion

#endif //#ifndef _CNDEFINE_H_

//=============================================================================
// END OF FILE
//=============================================================================