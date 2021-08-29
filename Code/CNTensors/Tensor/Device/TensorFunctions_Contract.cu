//=============================================================================
// FILENAME : TensorFunctions_Contract.cu
// 
// DESCRIPTION:
//
// Now, we still have no idea what kind of tensor we will confront, a high order tensor? or a large dimension tensor?
// So, the contract is implemented in a most general approach
//
// REVISION:
//  [09/12/2020 nbale]
//=============================================================================
#include "CNTensorsPch.h"


__BEGIN_NAMESPACE

#pragma region kernels

/**
 * (block * thread).x is index of left tensor, (block * thread).y is index of right tensor.
 */
template <class Tresult, class TLeft, class TRight> 
__global__ void _kernel_MM(Tresult* dest, const TLeft* __restrict__ left, const TRight* __restrict__ right,
    UINT contractDim,
    BYTE leftOrder, BYTE leftOrderToContract, const UINT* __restrict__ leftDim,
    BYTE rightOrder, BYTE rightOrderToContract, const UINT* __restrict__ rightDim,
    BYTE targetOrder, BYTE targetDim)
{
    const UINT uiThreadIdxLeft = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT uiThreadIdxRight = threadIdx.y + blockIdx.y * blockDim.y;

    Tresult res;
    _Zero(res);

    for (UINT i = 0; i < contractDim; ++i)
    {
        UINT uiLeftIdx;
        UINT uiRightIdx;
        res = _Add(res, _Mul(left[uiLeftIdx], right[uiRightIdx]));
    }


}

#pragma endregion

template <class Tresult, class TLeft, class TRight> __DLL_EXPORT
void MM(Tresult* dest, const TLeft* __restrict__ left, const TRight* __restrict__ right,
    BYTE leftOrder, BYTE leftOrderToContract, const UINT* __restrict__ leftDim,
    BYTE rightOrder, BYTE rightOrderToContract, const UINT* __restrict__ rightDim)
{
    
}

__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
