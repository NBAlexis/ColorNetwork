//=============================================================================
// FILENAME : CLGTest.cpp
// 
// DESCRIPTION:
//
// REVISION:
//  [12/2/2018 nbale]
//=============================================================================

#include "CNTest.h"

TestList* _testSuits;
#if Notyet

UINT RunTest(CParameters&params, TestList* pTest)
{
    appGeneral("\n=========== Testing:%s \n", pTest->m_sParamName);
    CParameters paramForTheTest = params.GetParameter(pTest->m_sParamName);
    appGeneral(_T("============= Parameters =============\n"));
    paramForTheTest.Dump(_T(""));
    //Initial
    if (!appInitialCLG(paramForTheTest))
    {
        return 1;
    }

    //Do the work
    CTimer timer;
    timer.Start();
    const UINT uiErrors = (*pTest->m_pfTest)(paramForTheTest);
    timer.Stop();
    appGeneral(_T("=========== Finished, errors: %d, cost: %f(ms)\n ------------- End --------------\n\n"), uiErrors, timer.Elapsed());

    //Final
    appQuitCLG();

    return uiErrors;
}

void ListAllTests(const THashMap<CCString, TArray<TestList*>*>& category)
{
    TArray<CCString> sKeys = category.GetAllKeys();
    for (INT k = 0; k < sKeys.Num(); ++k)
    {
        COUT << _T("============== ") << sKeys[k] << _T(" ==============\n");
        TArray<TestList*>* lst = category.GetAt(sKeys[k]); //category[] only work with non-const THashMap
        for (INT i = 0; i <= lst->Num() / 3; ++i)
        {
            for (INT j = 0; j < 3; ++j)
            {
                const INT indexOfTest = i * 3 + j;
                if (indexOfTest < lst->Num())
                {
                    TCHAR names[256];
                    appSprintf(names, 256, _T("%d - %s,    "), lst->GetAt(indexOfTest)->m_uiIndex, lst->GetAt(indexOfTest)->m_sParamName);
                    COUT << names;
                }
            }
            COUT << std::endl;
        }
    }
}

void DeleteAllLists(THashMap<CCString, TArray<TestList*>*>& category)
{
    //delete the lists
    TArray<CCString> sKeys = category.GetAllKeys();
    for (INT i = 0; i < sKeys.Num(); ++i)
    {
        appSafeDelete(category[sKeys[i]]);
    }
}
#endif


int main(int argc, char * argv[])
{
    appInitialTracer(PARANOIAC);

    appInitialCNLib(_T(""));
    CNHostTensor<_SComplex> tensor1;
    CNHostTensor<FLOAT> tensor2;
    UINT lengths[] = { 4, 4, 4, 4 };
    tensor1.CreateEmpty(lengths, 4);
    tensor2.CreateEmpty(lengths, 4);

    //tensor1.DebugPrint(16, 16);

    UINT strides[] = { 64, 16, 4, 1 };
    CNDeviceTensorCommonNaive calc;
    tensor1.Zero(calc, 0, strides, lengths, 4);

    tensor1.DebugPrint(16, 16);

    tensor1.One(calc, 0, strides, lengths, 4);

    tensor1.DebugPrint(16, 16);

    tensor1.Set(calc, 0.1f, 0, strides, lengths, 4);

    tensor1.DebugPrint(16, 16);

    tensor1.Sin(calc, 0, strides, lengths, 4);

    tensor1.DebugPrint(16, 16);

    tensor1.Add(calc, 1.0f, 0, strides, lengths, 4);

    tensor1.DebugPrint(16, 16);

    tensor2.Set(calc, tensor1, 0, strides, 0, strides, lengths, 4);

    tensor2.DebugPrint(16, 16);

    tensor2.Add(calc, tensor1, 0, strides, 0, strides, lengths, 4);

    tensor2.DebugPrint(16, 16);

    CCudaHelper::DebugFunction();
    CCudaHelper::DebugFunction();
    CCudaHelper::DebugFunction();

    appExitCNLib();

    return 0;
}

//=============================================================================
// END OF FILE
//=============================================================================
