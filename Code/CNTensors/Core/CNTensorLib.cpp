//=============================================================================
// FILENAME : CNTensorLib.cpp
// 
// DESCRIPTION:
//
//
// REVISION:
//  [31/05/2020 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

CNAPI CNTensorLib GCNLib;

CNTensorLib::CNTensorLib()
    : m_pCuda(NULL)
    , m_pOpWorkingSpace(NULL)
    , m_pRandom(NULL)
    , m_pDeviceRandom(NULL)
{
    
}

CNTensorLib::~CNTensorLib()
{
    
}

UBOOL CNTensorLib::Initial(const TCHAR* sConfigFile)
{
    CParameters params;
    CYAMLParser::ParseFile(sConfigFile, params);
    return Initial(params);
}

UBOOL CNTensorLib::Initial(CParameters& sConfig)
{
    m_pCuda = new CCudaHelper();
    m_pOpWorkingSpace = new CTensorOpWorkingSpace();
    CParameters paramRandom;
    if (sConfig.FetchParameterValue(_T("Random"), paramRandom))
    {
        InitialRandom(paramRandom);
    }
    
    return TRUE;
}

#pragma region Initial

void CNTensorLib::InitialRandom(CParameters& sConfig)
{
    CCString sRandom = _T("ER_Schrage");
    sConfig.FetchStringValue(_T("ERandom"), sRandom);
    ERandom eRandom = __STRING_TO_ENUM(ERandom, sRandom);

    CCString sSeed = _T("Timestamp");
    sConfig.FetchStringValue(_T("Seed"), sSeed);
    UINT uiSeed = 0;
    if (sSeed == _T("Timestamp"))
    {
        uiSeed = appGetTimeStamp();
    }
    else
    {
        uiSeed = appStoUI(sSeed.c_str());
    }

    m_pRandom = new CRandom(uiSeed, MAX_THREAD, eRandom);
    appCudaMalloc((void**)&m_pDeviceRandom, sizeof(CRandom));
    checkCudaErrors(cudaMemcpy(m_pDeviceRandom, m_pRandom, sizeof(CRandom), cudaMemcpyHostToDevice));
    m_pCuda->CopyRandomPointer(m_pDeviceRandom);
}

#pragma endregion

void CNTensorLib::Exit()
{
    appCudaFree(m_pDeviceRandom);
    appSafeDelete(m_pRandom);
    appSafeDelete(m_pOpWorkingSpace);
    appSafeDelete(m_pCuda);
    appFlushLog();

    //Do not call reset unless you are sure there is no more cudaFree
    //appCudaDeviceReset();
}

__END_NAMESPACE



//=============================================================================
// END OF FILE
//=============================================================================
