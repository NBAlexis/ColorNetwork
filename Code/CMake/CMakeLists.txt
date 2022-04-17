cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if (DEFINED NVCCROOT)
    set(CMAKE_CUDA_COMPILER ${NVCCROOT})
    MESSAGE("CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
endif()

set(CUDA_CMP "compute_61")
set(CUDA_SM "sm_61")
if (DEFINED CUDASM)
    set(CUDA_CMP "compute_${CUDASM}")
    set(CUDA_SM "sm_${CUDASM}")
endif()

MESSAGE("Note: arch is ${CUDA_CMP} and ${CUDA_SM}.")
MESSAGE("52 for 970, 61 for GTX10, 70 for V100, 75 for RTX20, 86 for RTX30")
project(ColorNetwork LANGUAGES CXX CUDA)

set(CMAKE_GENERATOR_PLATFORM x64)

# We start from CMAKE_SOURCE_DIR which should be /Code/CMake
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_SOURCE_DIR}/../../Bin/Ubuntu)
set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH  ${CMAKE_CURRENT_BINARY_DIR})
# This is our code file dir
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/..)
# Flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options=/bigobj")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Wno-unknown-pragmas -Wno-strict-overflow -Wno-class-memaccess")
add_definitions(-D_UBUNTU)
MESSAGE("CMAKE_CUDA_FLAGS flag = ${CMAKE_CUDA_FLAGS}")
MESSAGE("CMAKE_CXX_FLAGS flag = ${CMAKE_CXX_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR}/CNTensors)
add_library(CNTensors STATIC
    ${PROJECT_SOURCE_DIR}/CNTensors/Core/CNDefine.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Core/CNFloat.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Core/CNSetup.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Core/CNTensorLib.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Core/CudaIncludes.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CudaComplexFunction.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CudaHelper.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CudaHelperFunctions.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CNFFT.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CNRandom.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Platform/PlatformDefine.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Platform/PlatformIncs.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CNTensors.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CNTensorsPch.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CNDeviceTensor.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CNDeviceTensorCalculatorGather.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CNDeviceTensorCommon.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CNDeviceTensorElementOperator.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/Common/CNDeviceTensorCommonNaive.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CTensorOpWorkingSpace.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/TensorFunctions.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Host/CNHostTensor.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/CYAMLParser.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/CBitFlag.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/CCString.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/CNMD5.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/CLinkedList.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/MemStack.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/STDStringFunctions.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/TArray.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/TemplateFunctions.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/THashMap.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/EnumGather.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Timer.h
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Tracer.h
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CNFFT.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CNRandom.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/CudaFunctions/CudaHelper.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CNDeviceTensorCommon.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CNDeviceTensorElementOperator.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/Common/CNDeviceTensorCommonNaive.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/Common/CNDeviceTensorCommonNaiveAdd.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/TensorFunctions_Common.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/TensorFunctions_Contract.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Device/CTensorOpWorkingSpace.cu
    ${PROJECT_SOURCE_DIR}/CNTensors/CNTensorsPch.cpp
    ${PROJECT_SOURCE_DIR}/CNTensors/Core/CNTensorLib.cpp
    ${PROJECT_SOURCE_DIR}/CNTensors/Tensor/Host/CNHostTensor.cpp
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/CYAMLParser.cpp
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Data/CCString.cpp
    ${PROJECT_SOURCE_DIR}/CNTensors/Tools/Tracer.cpp
    )

# Request that CNTensors be built with -std=c++14
# As this is a public compile feature anything that links to 
# CNTensors will also build with -std=c++14
target_compile_features(CNTensors PUBLIC cxx_std_14)
 
# We need to explicitly state that we need all CUDA files in the 
# CNTensors library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(CNTensors PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CNTensors -lcurand)
target_link_libraries(CNTensors -lcufft)

# To enable the double, the minimum arch is 6.0
target_compile_options(CNTensors PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=${CUDA_CMP},code=${CUDA_SM}>)




# ==================== 
# CNTest 
# =================

include_directories(${PROJECT_SOURCE_DIR}/CNTest)
add_executable(CNTest 
    ${PROJECT_SOURCE_DIR}/CNTest/CNTest.h
    ${PROJECT_SOURCE_DIR}/CNTest/CNTest.cpp
    )

target_compile_features(CNTest PUBLIC cxx_std_14)
target_link_libraries(CNTest CNTensors)

