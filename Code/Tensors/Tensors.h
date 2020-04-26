//=============================================================================
// FILENAME : Tensors.h
// 
// DESCRIPTION:
// This is the one header file for all
//
// REVISION:
//  [24/4/2020 nbalexis]
//=============================================================================
#pragma once

#ifndef _TENSORS_H_
#define _TENSORS_H_

#include "Core/CNSetup.h"
#include "Core/CNDefine.h"

#if defined(_CN_WIN)
#   if !defined(CNAPI)
#        define __LIB_TITLE__    "Tensors"
#       ifdef _CN_PRIVATE
#           define CNAPI __DLL_EXPORT
#       else
#           define CNAPI __DLL_IMPORT
#       endif
#       ifndef _CN_PRIVATE
#           ifdef _CN_DEBUG
#                define __LIB_FILE__ __LIB_TITLE__ "_d.lib"
#            else
#                define __LIB_FILE__ __LIB_TITLE__ ".lib"
#            endif
#            pragma __IMPORT_LIB(__LIB_FILE__)
#            pragma message("linking with " __LIB_FILE__ "...")
#            undef __LIB_FILE__
#            undef __LIB_TITLE__
#       endif
#   endif
#else
#    define CNAPI  
#endif

#include "Platform/PlatformIncs.h"
#include "Platform/PlatformDefine.h"

#include "Tools/Data/STDStringFunctions.h"
#include "CudaFunctions/CudaHelperFunctions.h"
#include "Core/CNFloat.h"
#include "Tools/Data/CLinkedList.h"
#include "Tools/Data/TemplateFunctions.h"
#include "Tools/Data/MemStack.h"
#include "Tools/Data/CBitFlag.h"
//using these class to avoid warnings by STL...
#include "Tools/Data/TArray.h"
#include "Tools/Data/CCString.h"
#include "Tools/Data/THashMap.h"
#include "Tools/EnumGather.h"
#include "Tools/Tracer.h"

#include "CudaFunctions/CNRandom.h"
#include "CudaFunctions/CNFFT.h"

#ifndef _CN_PRIVATE
__USE_NAMESPACE
#endif

#endif