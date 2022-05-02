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

void CNTensorLib::Initial(const TCHAR* sConfigFile)
{
    m_pCuda = new CCudaHelper();
    m_pOpWorkingSpace = new CTensorOpWorkingSpace();

    InitialRandom();
}

#pragma region Initial

void CNTensorLib::InitialRandom()
{
    m_pRandom = new CRandom(0, MAX_THREAD, ER_Schrage);
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
