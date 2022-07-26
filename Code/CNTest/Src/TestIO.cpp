//=============================================================================
// FILENAME : TestIO.cpp
// 
// DESCRIPTION:
//
//     Test the operations on tensors
//
// REVISION[d-m-y]:
//  [26/07/2022 nbale]
//=============================================================================

#include "CNTest.h"

#define TESTFILE "test.cnt"

template <class T1, class T2>
UINT TestIOFunction(CParameters& param, UINT uiRandomType)
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
    CNIndexBlock block(names);
    CNHostTensor<T1> totest;
    CNDeviceTensorCommonNaive common_calc;
    CNDeviceTensorContractionNaive contract_calc;
    totest.Create(tobecreate);
    totest.Random(common_calc, uiRandomType);
    //totest.PrintContent(lengths[0]);
    if (!totest.SaveToFile(TESTFILE))
    {
        appGeneral(_T("Save file failed: %s\n"), TESTFILE);
        totest.Release();
        return 1;
    }

    CNHostTensor<T2> totest2;
    if (!totest2.CreateWithFile(TESTFILE))
    {
        appGeneral(_T("Load file failed: %s\n"), TESTFILE);
        totest.Release();
        totest2.Release();
        return 1;
    }
    totest.Sub(common_calc, block, totest2, block);

    CNHostTensor<T1>* pPooledCopy = totest.GetPooledCopy();
    T1 summed = pPooledCopy->ReduceSum(contract_calc);
    TOperator_Set<FLOAT, T1> setopfc;
    FLOAT fSummed;
    setopfc.Do(fSummed, summed);
    appGeneral(_T("a = %f (expecting: 0)\n"), fSummed / totest.GetVolume());

    pPooledCopy->ReleaseMe();
    
    //totest2.PrintContent(lengths[0]);

    totest.Release();
    totest2.Release();

    if (abs(fSummed) > 0.001f)
    {
        return 1;
    }

    return 0;
}

UINT TestIOFunctionComp(CParameters& param)
{
    return TestIOFunction<_SComplex, _DComplex>(param, 0);
}

UINT TestIOFunctionInteger(CParameters& param)
{
    return TestIOFunction<INT, FLOAT>(param, 1);
}

__REGIST_TEST(TestIOFunctionComp, IO, SaveLoadTestComp);

__REGIST_TEST(TestIOFunctionInteger, IO, SaveLoadTestInt);

//=============================================================================
// END OF FILE
//=============================================================================
