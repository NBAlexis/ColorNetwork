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

#ifndef _CNTENSORS_H_
#define _CNTENSORS_H_

#include "Core/CNSetup.h"
#include "Core/CNDefine.h"

#if defined(_CN_WIN)
#   if !defined(CNAPI)
#        define __LIB_TITLE__    "CNTensors"
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
#include "Core/CudaIncludes.h"
#include "Core/CNFloat.h"
#include "Tools/Data/CLinkedList.h"
#include "Tools/Data/TemplateFunctions.h"
#include "Tools/Data/MemStack.h"
#include "Tools/Data/CBitFlag.h"
//using these class to avoid warnings by STL..., see: https://code.i-harness.com/en/q/56642a
#include "Tools/Data/TArray.h"
#include "Tools/Data/CCString.h"
#include "Tools/Data/THashMap.h"
#include "Tools/Data/CNMD5.h"
#include "Tools/EnumGather.h"
#include "Tools/Tracer.h"
#include "Tools/Timer.h"
#include "Tools/CYAMLParser.h"
#include "Core/CNTensorLib.h"

#include "CudaFunctions/CudaHelperFunctions.h"
#include "CudaFunctions/CudaComplexFunction.h"
#include "CudaFunctions/CudaHelper.h"
#include "CudaFunctions/CNRandom.h"
#include "CudaFunctions/CNFFT.h"

//====================== Tensor =======================
#include "Tensor/Device/CNDeviceTensorElementOperator.h"
#include "Tensor/Device/CTensorOpWorkingSpace.h"
#include "Tensor/Device/TensorFunctions.h"
#include "Tensor/Device/CNDeviceTensorCommon.h"
#include "Tensor/Device/CNDeviceTensorContraction.h"
#include "Tensor/Device/CNDeviceTensorDecompose.h"
#include "Tensor/Device/CNDeviceTensor.h"
#include "Tensor/Host/CNOneIndex.h"
#include "Tensor/Host/CNIndex.h"
#include "Tensor/Host/CNHostTensor.h"

//====================== Caclculator =======================
#include "Tensor/Device/Common/CNDeviceTensorNaiveIndexMapping.h"
#include "Tensor/Device/Common/CNDeviceTensorRandom.h"
#include "Tensor/Device/Common/CNDeviceTensorCommonNaive.h"
#include "Tensor/Device/Contraction/CNDeviceTensorContractionNaive.h"

//====================== Decompose =======================
#include "Tensor/Host/Decompose/CNMatrixSVD.h"


//#include "Tensor/CNIndex.h"

#ifndef _CN_PRIVATE
__USE_NAMESPACE

inline CCString GetCNVersion()
{
    CCString sRet;
    sRet.Format(_T("%d.%d"), __GVERSION, __GVERSION_S);
    return sRet;
}

#endif

#endif