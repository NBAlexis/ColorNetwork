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
{
    
}

CNTensorLib::~CNTensorLib()
{
    
}

void CNTensorLib::Initial(const class CCString& sConfigFile)
{
    m_pCuda = new CCudaHelper();
}

void CNTensorLib::Exit()
{
    appSafeDelete(m_pCuda);

    appFlushLog();
}

__END_NAMESPACE



//=============================================================================
// END OF FILE
//=============================================================================
