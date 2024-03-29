//=============================================================================
// FILENAME : CNTest.cpp
// 
// DESCRIPTION:
//
// REVISION[d-m-y]:
//  [13/05/2022 nbale]
//=============================================================================

#include "CNTest.h"

TestList* _testSuits;

UINT RunTest(CParameters&params, TestList* pTest)
{
    appGeneral("\n=========== Testing:%s \n", pTest->m_sParamName);
    CParameters paramForTheTest = params.GetParameter(pTest->m_sParamName);
    appGeneral(_T("============= Parameters =============\n"));
    paramForTheTest.Dump(_T(""));
    //Initial
    if (!appInitialCNLib(paramForTheTest))
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
    appExitCNLib();

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


int main(int argc, char* argv[])
{
    //Load settings
    CParameters params;
    CYAMLParser::ParseFile(_T("../YAMLs/TensorBasicOps.yaml"), params);
    CYAMLParser::ParseFile(_T("../YAMLs/TensorIOTest.yaml"), params);
    CNSetupLog(params);

    TArray<TestList*> allTests;
    THashMap<CCString, TArray<TestList*>*> category;
    UINT uiIndex = 0;
    for (TestList* pTest = _testSuits; NULL != pTest; pTest = pTest->m_pNext)
    {
        if (params.Exist(pTest->m_sParamName))
        {
            pTest->m_uiIndex = uiIndex;
            ++uiIndex;
            allTests.AddItem(pTest);
            CCString sCategory = pTest->m_sCatogary;
            if (category.Exist(sCategory))
            {
                category[sCategory]->AddItem(pTest);
            }
            else
            {
                TArray<TestList*>* newList = new TArray<TestList*>();
                newList->AddItem(pTest);
                category.SetAt(sCategory, newList);
            }
        }
    }

    //INT inputNumber = -1;
    ListAllTests(category);
    while (TRUE)
    {
        COUT << _T("============== CLG v") << GetCNVersion().c_str() << _T("==============\nq - Quit,  l - List all,  r - Run all,  p - Print Device info\n");
        //ListAllTests(category);
        //inputNumber = -1;
        std::string name;
        std::getline(std::cin, name);
        CCString sRes(name.c_str());
        INT number = appStrToINT(sRes);
        UBOOL bExcuted = FALSE;
        sRes.MakeLower();
        if (sRes == _T("q"))
        {
            break;
        }

        if (sRes == _T("l"))
        {
            ListAllTests(category);
            bExcuted = TRUE;
        }
        else if (sRes == _T("p"))
        {
            CCudaHelper::DeviceQuery();
            bExcuted = TRUE;
        }
        else if (appIntToString(number) == sRes)
        {
            if (number >= 0 && number < allTests.Num())
            {
                RunTest(params, allTests[number]);
                bExcuted = TRUE;
            }
        }
        else if (sRes == _T("r"))
        {
            CTimer timer;
            timer.Start();
            UINT uiError = 0;
            UINT uiPassed = 0;
            TArray<CCString> problemTest;
            for (INT i = 0; i < allTests.Num(); ++i)
            {
                UINT uiThisError = RunTest(params, allTests[i]);
                if (0 == uiThisError)
                {
                    ++uiPassed;
                }
                else
                {
                    uiError += uiThisError;
                    problemTest.AddItem(allTests[i]->m_sParamName);
                }
            }
            timer.Stop();
            appGeneral(_T("\nRun all test with %d(success) / %d(total) (with %d errors) and %f secs\n\n\n================\n"), uiPassed, allTests.Num(), uiError, timer.Elapsed() * 0.001f);
            for (INT i = 0; i < problemTest.Num(); ++i)
            {
                appGeneral(_T("problem test suits: %s\n"), problemTest[i].c_str());
            }
            bExcuted = TRUE;
        }
        else if (sRes == _T("class"))
        {
            //not yet nessesary
            //GClassGather.TraceAllClass();
            appGeneral(_T("\n================================\n"));
        }
        else
        {
            TArray<CCString> keys = category.GetAllKeys();
            for (INT i = 0; i < keys.Num(); ++i)
            {
                CCString sInput = sRes;
                CCString sKey = keys[i];
                sKey.MakeLower();
                if (sInput == sKey)
                {
                    CTimer timer;
                    timer.Start();
                    UINT uiError = 0;
                    UINT uiPassed = 0;
                    TArray<CCString> unpassed;
                    for (INT j = 0; j < category[keys[i]]->Num(); ++j)
                    {
                        UINT uiThisError = RunTest(params, category[keys[i]]->GetAt(j));
                        if (0 == uiThisError)
                        {
                            ++uiPassed;
                        }
                        else
                        {
                            uiError += uiThisError;
                            unpassed.AddItem(category[keys[i]]->GetAt(j)->m_sParamName);
                        }
                    }
                    timer.Stop();
                    appGeneral(_T("Run all %s test with %d(success) / %d(total) (with %d errors) and %f secs\n\n\n================\n"),
                        keys[i].c_str(), uiPassed, category[keys[i]]->Num(),
                        uiError, timer.Elapsed() * 0.001f);
                    for (INT unpassidx = 0; unpassidx < unpassed.Num(); ++unpassidx)
                    {
                        appGeneral(_T("Faield:%s\n"), unpassed[unpassidx].c_str());
                    }
                    break;
                    //bExcuted = TRUE;
                }
            }
        }

        if (!bExcuted)
        {
            COUT << _T("Input commond:") << name << _T(" not kown") << std::endl;
        }
    }
    DeleteAllLists(category);
    return 0;
}

//=============================================================================
// END OF FILE
//=============================================================================
