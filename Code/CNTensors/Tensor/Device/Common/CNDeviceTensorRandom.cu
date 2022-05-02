//=============================================================================
// FILENAME : CNDeviceTensorCommonNaive.cu
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [19/06/2021 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

__OVER_ALL_TYPE_ONE(__IMPLEMENT_COMMON_RANDOM)

#pragma region kernels

template<class T>
__global__ void _CN_LAUNCH_BOUND
_kernel_CreateRandom(T* buffer, UINT iCase)
{
	UINT uiThreadIdx = threadIdx.x;
	UINT uiTensorIdx = threadIdx.x + blockIdx.x * blockDim.x;
	switch (iCase)
	{
	case 0:
		buffer[uiTensorIdx] = static_cast<T>(_deviceRandomF(uiThreadIdx));
		break;
	case 1:
		buffer[uiTensorIdx] = static_cast<T>(_deviceRandomGaussF(uiThreadIdx));
		break;
	default:
		buffer[uiTensorIdx] = static_cast<T>(_deviceRandomIF(uiThreadIdx, iCase));
		break;
	}
}

template<>
__global__ void _CN_LAUNCH_BOUND
_kernel_CreateRandom(_SComplex* buffer, UINT iCase)
{
	UINT uiThreadIdx = threadIdx.x;
	UINT uiTensorIdx = threadIdx.x + blockIdx.x * blockDim.x;
	switch (iCase)
	{
	case 0:
		buffer[uiTensorIdx] = _deviceRandomC(uiThreadIdx);
		break;
	case 1:
		buffer[uiTensorIdx] = _deviceRandomGaussC(uiThreadIdx);
		break;
	default:
		buffer[uiTensorIdx] = _deviceRandomZN(uiThreadIdx, iCase);
		break;
	}
}

template<>
__global__ void _CN_LAUNCH_BOUND
_kernel_CreateRandom(_DComplex* buffer, UINT iCase)
{
	UINT uiThreadIdx = threadIdx.x;
	UINT uiTensorIdx = threadIdx.x + blockIdx.x * blockDim.x;
	switch (iCase)
	{
	case 0:
		buffer[uiTensorIdx] = _stod(_deviceRandomC(uiThreadIdx));
		break;
	case 1:
		buffer[uiTensorIdx] = _stod(_deviceRandomGaussC(uiThreadIdx));
		break;
	default:
		buffer[uiTensorIdx] = _stod(_deviceRandomZN(uiThreadIdx, iCase));
		break;
	}
}

template<class T>
__global__ void _CN_LAUNCH_BOUND
_kernel_RandomBlocked(T* buffer, UINT iCase,
	const UINT* __restrict__ srcStride,
	UINT srcIndexStart,
	const UINT* __restrict__ mutipliedlengths,
	BYTE byIndexCount)
{
	UINT uiThreadIdx = threadIdx.x;
	const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
	const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);

	switch (iCase)
	{
	case 0:
		buffer[uiIdxSrc] = static_cast<T>(_deviceRandomF(uiThreadIdx));
		break;
	case 1:
		buffer[uiIdxSrc] = static_cast<T>(_deviceRandomGaussF(uiThreadIdx));
		break;
	default:
		buffer[uiIdxSrc] = static_cast<T>(_deviceRandomIF(uiThreadIdx, iCase));
		break;
	}
}

template<>
__global__ void _CN_LAUNCH_BOUND
_kernel_RandomBlocked(_SComplex* buffer, UINT iCase,
	const UINT* __restrict__ srcStride,
	UINT srcIndexStart,
	const UINT* __restrict__ mutipliedlengths,
	BYTE byIndexCount)
{
	UINT uiThreadIdx = threadIdx.x;
	const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
	const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);

	switch (iCase)
	{
	case 0:
		buffer[uiIdxSrc] = _deviceRandomC(uiThreadIdx);
		break;
	case 1:
		buffer[uiIdxSrc] = _deviceRandomGaussC(uiThreadIdx);
		break;
	default:
		buffer[uiIdxSrc] = _deviceRandomZN(uiThreadIdx, iCase);
		break;
	}
}

template<>
__global__ void _CN_LAUNCH_BOUND
_kernel_RandomBlocked(_DComplex* buffer, UINT iCase,
	const UINT* __restrict__ srcStride,
	UINT srcIndexStart,
	const UINT* __restrict__ mutipliedlengths,
	BYTE byIndexCount)
{
	UINT uiThreadIdx = threadIdx.x;
	const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
	const UINT uiIdxSrc = _deviceThreadIdxToTensorIdxNaive(srcStride, srcIndexStart, mutipliedlengths, uiIdx, byIndexCount);

	switch (iCase)
	{
	case 0:
		buffer[uiIdxSrc] = _stod(_deviceRandomC(uiThreadIdx));
		break;
	case 1:
		buffer[uiIdxSrc] = _stod(_deviceRandomGaussC(uiThreadIdx));
		break;
	default:
		buffer[uiIdxSrc] = _stod(_deviceRandomZN(uiThreadIdx, iCase));
		break;
	}
}

template<class T>
__global__ void _CN_LAUNCH_BOUND
_kernel_RandomBlocked_Small(T* buffer, UINT iCase,
	const UINT* __restrict__ srcStride,
	UINT srcIndexStart,
	const UINT* __restrict__ mutipliedlengths,
	BYTE byIndexCount)
{
	UINT uiThreadIdx = threadIdx.x;
	const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
	const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
	const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);

	switch (iCase)
	{
	case 0:
		buffer[uiIdxSrc] = static_cast<T>(_deviceRandomF(uiThreadIdx));
		break;
	case 1:
		buffer[uiIdxSrc] = static_cast<T>(_deviceRandomGaussF(uiThreadIdx));
		break;
	default:
		buffer[uiIdxSrc] = static_cast<T>(_deviceRandomIF(uiThreadIdx, iCase));
		break;
	}
}

template<>
__global__ void _CN_LAUNCH_BOUND
_kernel_RandomBlocked_Small(_SComplex* buffer, UINT iCase,
	const UINT* __restrict__ srcStride,
	UINT srcIndexStart,
	const UINT* __restrict__ mutipliedlengths,
	BYTE byIndexCount)
{
	UINT uiThreadIdx = threadIdx.x;
	const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
	const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
	const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);

	switch (iCase)
	{
	case 0:
		buffer[uiIdxSrc] = _deviceRandomC(uiThreadIdx);
		break;
	case 1:
		buffer[uiIdxSrc] = _deviceRandomGaussC(uiThreadIdx);
		break;
	default:
		buffer[uiIdxSrc] = _deviceRandomZN(uiThreadIdx, iCase);
		break;
	}
}

template<>
__global__ void _CN_LAUNCH_BOUND
_kernel_RandomBlocked_Small(_DComplex* buffer, UINT iCase,
	const UINT* __restrict__ srcStride,
	UINT srcIndexStart,
	const UINT* __restrict__ mutipliedlengths,
	BYTE byIndexCount)
{
	UINT uiThreadIdx = threadIdx.x;
	const UINT uiIdx = threadIdx.x + blockIdx.x * blockDim.x;
	const SWorkingIndex idx = _deviceThreadIndexToWorkIndexNavie(uiIdx, mutipliedlengths, byIndexCount);
	const UINT uiIdxSrc = _deviceWorkIndexToTensorIndexNaive(idx.m_Idx, srcStride, srcIndexStart, byIndexCount);

	switch (iCase)
	{
	case 0:
		buffer[uiIdxSrc] = _stod(_deviceRandomC(uiThreadIdx));
		break;
	case 1:
		buffer[uiIdxSrc] = _stod(_deviceRandomGaussC(uiThreadIdx));
		break;
	default:
		buffer[uiIdxSrc] = _stod(_deviceRandomZN(uiThreadIdx, iCase));
		break;
	}
}

#pragma endregion

template<class T>
void TCNDeviceTensorRandom<T>::Random(T* pBuffer, UINT uiRandomType, UINT uiTotalSize)
{
	UINT block = 1;
	UINT thread = 1;
	GetDecompose(BOUND_THREAD, uiTotalSize, block, thread);
	_kernel_CreateRandom << <block, thread >> > (pBuffer, uiRandomType);
}

template<class T>
void TCNDeviceTensorRandom<T>::Random(
	T* pBuffer,
	UINT uiRandomType,
	UINT dstIndexStart,
	const UINT* __restrict__ dstStride,
	const UINT* __restrict__ lengths,
	BYTE byIndexCount)
{
	const UINT dataSize = sizeof(UINT) * byIndexCount;
	const UINT totalBufferSize = dataSize * 2;
	UINT uiBlock, uiThread;
	SimpleThreadDecompose(lengths, byIndexCount, uiBlock, uiThread);

	BYTE* deviceBuffer = appGetSmallDeviceBuffer(totalBufferSize);

	UINT* hostBuffer = (UINT*)appAlloca(dataSize);
	_memcpy_hd(deviceBuffer, dstStride, dataSize);
	__BuildMultiplyLength(deviceBuffer + dataSize);

	__KERNALCALNAIVE(_kernel_RandomBlocked,
		pBuffer,
		uiRandomType,
		(UINT*)deviceBuffer,
		dstIndexStart,
		(UINT*)(deviceBuffer + dataSize),
		byIndexCount
	);
}


__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
