//=============================================================================
// FILENAME : CNHostTensor.cpp
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [10/31/2021 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

template class CNHostTensor<BYTE>;
template class CNHostTensor<USHORT>;
template class CNHostTensor<UINT>;
template class CNHostTensor<QWORD>;
template class CNHostTensor<SBYTE>;
template class CNHostTensor<SHORT>;
template class CNHostTensor<INT>;
template class CNHostTensor<SQWORD>;
template class CNHostTensor<FLOAT>;
template class CNHostTensor<DOUBLE>;
template class CNHostTensor<_SComplex>;
template class CNHostTensor<_DComplex>;

//template<class T>
//void CNHostTensor<T>::Zero(ECalculator eCalc, UINT uiIndexStart, const UINT* strides, const UINT* lengths, BYTE uiIndexCount)
//{
//    //TOperator_Zero<T> op;
//    //TOperator_Zero<T> opone;
//    //switch (eCalc)
//    //{
//    //    
//    //}
//    //GetCommonCalculator(eCalc)->OneOperator(opone, &m_cDeviceTensor, uiIndexStart, strides, lengths, uiIndexCount);
//    //OneOperator<TOperator_Zero<T>>(eCalc, uiIndexStart, strides, lengths, uiIndexCount);
//    Calc_Zero(eCalc, &m_cDeviceTensor, uiIndexStart, strides, lengths, uiIndexCount);
//}
//
//template<class T>
//void CNHostTensor<T>::One(ECalculator eCalc, UINT uiIndexStart, const UINT* strides, const UINT* lengths, BYTE uiIndexCount)
//{
//    //TOperator_Zero<T> op;
//    //TOperator_Zero<T> opone;
//    //switch (eCalc)
//    //{
//    //    
//    //}
//    //GetCommonCalculator(eCalc)->OneOperator(opone, &m_cDeviceTensor, uiIndexStart, strides, lengths, uiIndexCount);
//    //OneOperator<TOperator_Zero<T>>(eCalc, uiIndexStart, strides, lengths, uiIndexCount);
//    Calc_One(eCalc, &m_cDeviceTensor, uiIndexStart, strides, lengths, uiIndexCount);
//}

__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
