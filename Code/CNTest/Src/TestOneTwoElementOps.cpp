//=============================================================================
// FILENAME : TestOneTwoLelemntOps.cpp
// 
// DESCRIPTION:
//
//     Test the operations on tensors
//
// REVISION[d-m-y]:
//  [13/05/2022 nbale]
//=============================================================================

#include "CNTest.h"

/**
* a = 1
* suma = volume
* a = a * Pi * I / 2
* exp(a)
* a = a * I + 1
*/
template <class T>
UINT TestSimpleOperators(CParameters& param)
{
    UINT uiErrors = 0;

    TArray<UINT> lengths;
    TArray<CCString> names;
    param.FetchValueArrayUINT(_T("Lengths"), lengths);
    param.FetchStringVectorValue(_T("Names"), names);
    if (lengths.Num() != names.Num() || lengths.Num() < 1)
    {
        return 1;
    }

    CNIndex tobecreate(names, lengths);
    CNHostTensor<T>* totest = new CNHostTensor<T>();
    CNDeviceTensorCommonNaive common_calc;
    CNDeviceTensorContractionNaive contract_calc;
    totest->Create(tobecreate);
    totest->One(common_calc);

    CNHostTensor<T>* pPooledCopy = totest->GetPooledCopy();
    T summed  = pPooledCopy->ReduceSum(contract_calc);
    TOperator_Set<FLOAT, T> setopfc;
    FLOAT fSummed;
    setopfc.Do(fSummed, summed);
    appGeneral(_T("a = %f (expecting: 1)\n"), fSummed / totest->GetVolume());

    if (abs(fSummed - totest->GetVolume()) > 0.001f)
    {
        pPooledCopy->ReleaseMe();
        appSafeDelete(totest);
        return 1;
    }

    totest->Mul(common_calc, make_cuComplex(0.0f, PIF));
    totest->Div(common_calc, 2);
    totest->CopyTo(pPooledCopy);
    summed = pPooledCopy->ReduceSum(contract_calc);
    TOperator_Set<_SComplex, T> setopcc;
    _SComplex cSummed;
    setopcc.Do(cSummed, summed);
    appGeneral(_T("a = %f + %f I (expecting: 0 + 1.57 I)\n"), cSummed.x / totest->GetVolume(), cSummed.y / totest->GetVolume());

    if (abs(cSummed.x) > 0.001f)
    {
        pPooledCopy->ReleaseMe();
        appSafeDelete(totest);
        return 1;
    }

    if (abs(cSummed.y - PIF * totest->GetVolume() / 2.0f) > 0.001f)
    {
        pPooledCopy->ReleaseMe();
        appSafeDelete(totest);
        return 1;
    }

    totest->Exp(common_calc);
    totest->CopyTo(pPooledCopy);
    summed = pPooledCopy->ReduceSum(contract_calc);
    setopcc.Do(cSummed, summed);
    appGeneral(_T("a = %f + %f I (expecting 0 + I)\n"), cSummed.x / totest->GetVolume(), cSummed.y / totest->GetVolume());

    if (abs(cSummed.x) > 0.001f)
    {
        pPooledCopy->ReleaseMe();
        appSafeDelete(totest);
        return 1;
    }

    if (abs(cSummed.y - totest->GetVolume()) > 0.001f)
    {
        pPooledCopy->ReleaseMe();
        appSafeDelete(totest);
        return 1;
    }

    totest->Mul(common_calc, make_cuComplex(0.0f, 1.0f));
    totest->Add(common_calc, 1);
    totest->CopyTo(pPooledCopy);
    summed = pPooledCopy->ReduceSum(contract_calc);
    setopcc.Do(cSummed, summed);
    appGeneral(_T("a = %f + %f I (expecting 0)\n"), cSummed.x / totest->GetVolume(), cSummed.y / totest->GetVolume());

    if (abs(cSummed.x) > 0.001f)
    {
        pPooledCopy->ReleaseMe();
        appSafeDelete(totest);
        return 1;
    }

    if (abs(cSummed.y) > 0.001f)
    {
        pPooledCopy->ReleaseMe();
        appSafeDelete(totest);
        return 1;
    }

    pPooledCopy->ReleaseMe();
    appSafeDelete(totest);

    return uiErrors;
}

UINT ElementOpsSComp(CParameters& param)
{
    return TestSimpleOperators<_SComplex>(param);
}

UINT ElementOpsDComp(CParameters& param)
{
    return TestSimpleOperators<_DComplex>(param);
}

__REGIST_TEST(ElementOpsSComp, Common, ElementOpsSComp);

__REGIST_TEST(ElementOpsDComp, Common, ElementOpsDComp);


//=============================================================================
// END OF FILE
//=============================================================================
