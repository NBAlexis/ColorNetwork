//=============================================================================
// FILENAME : CNDeviceTensorCalculatorGather.h
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [11/01/2021 nbale]
//=============================================================================

#ifndef _CNDEVICETENSORCALCULATORGATHER_H_
#define _CNDEVICETENSORCALCULATORGATHER_H_

__BEGIN_NAMESPACE

/**
 * I find it difficult to put those as member functions in a template class
 * So, I just make those normal functions
 *
 * Note the implementation is not in .h file
 */
//template<class T, class Operator>
//void Calc_OneOperator(
//    ECalculator eCalc,
//    CNDeviceTensor<T>* tensor,
//    const TOperator_D<Operator, T>& op,
//    const UINT dstIndexStart,
//    const UINT* __restrict__ dstStride,
//    const UINT* __restrict__ lengths,
//    BYTE byIndexCount)
//{
//    switch (eCalc)
//    {
//    case EC_Naive:
//        CNDeviceTensorCommonNaive<T>().OneOperator(op, tensor, dstIndexStart, dstStride, lengths, byIndexCount);
//        break;
//    default:
//        appCrucial(_T("Calc_OneOperator:: Calculator not implemented!\n"));
//        break;
//    }
//}

#pragma region specific operators

template<class T>
void Calc_Zero(
    ECalculator eCalc,
    CNDeviceTensor<T>* tensor,
    const UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount)
{
    switch (eCalc)
    {
    case EC_Naive:
        CNDeviceTensorCommonNaiveOneOperator<TOperator_Zero<T>, T>(tensor->m_pDeviceDataBuffer)
        .OneOperator(dstIndexStart, dstStride, lengths, byIndexCount);
        break;
    default:
        appCrucial(_T("Calc_OneOperator:: Calculator not implemented!\n"));
        break;
    }
}

template<class T>
void Calc_One(
    ECalculator eCalc,
    CNDeviceTensor<T>* tensor,
    const UINT dstIndexStart,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ lengths,
    BYTE byIndexCount)
{
    switch (eCalc)
    {
    case EC_Naive:
        //CNDeviceTensorCommonNaive<T>(tensor->m_pDeviceDataBuffer).One(dstIndexStart, dstStride, lengths, byIndexCount);
        CNDeviceTensorCommonNaiveOneOperator<TOperator_One<T>, T>(tensor->m_pDeviceDataBuffer)
            .OneOperator(dstIndexStart, dstStride, lengths, byIndexCount);
        break;
    default:
        appCrucial(_T("Calc_OneOperator:: Calculator not implemented!\n"));
        break;
    }
}

#pragma endregion

__END_NAMESPACE

#endif //#ifndef _CNDEVICETENSORCALCULATORGATHER_H_


//=============================================================================
// END OF FILE
//=============================================================================
