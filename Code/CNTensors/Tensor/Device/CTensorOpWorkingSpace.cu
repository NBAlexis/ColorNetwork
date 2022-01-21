//=============================================================================
// FILENAME : CTensorOpWorkingSpace.cu
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [30/05/2020 nbale]
//=============================================================================

#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

CTensorOpWorkingSpace::CTensorOpWorkingSpace()
    : m_pSmallBuffer(NULL)
    , m_uiSmallBufferSize(0)
    , m_pDeviceZeroStart(NULL)
{
    appCudaMalloc((void**)&m_pDeviceZeroStart, sizeof(UINT) * _kMaxSupportedOrder);
    UINT zeros[_kMaxSupportedOrder];
    memset(zeros, 0, sizeof(UINT) * _kMaxSupportedOrder);
    _memcpy_hd(m_pDeviceZeroStart, zeros, sizeof(UINT) * _kMaxSupportedOrder);
}

CTensorOpWorkingSpace::~CTensorOpWorkingSpace()
{
    appCudaFree(m_pSmallBuffer);
    appCudaFree(m_pDeviceZeroStart);
}

BYTE* CTensorOpWorkingSpace::GetSmallDeviceBuffer(UINT uiSize)
{
    if (NULL != m_pSmallBuffer && m_uiSmallBufferSize < uiSize)
    {
        appCudaFree(m_pSmallBuffer);
    }
    if (NULL == m_pSmallBuffer)
    {
        m_uiSmallBufferSize = (uiSize + _kSmallBufferSize) & (~_kSmallBufferSize);
        appCudaMalloc((void**)&m_pSmallBuffer, m_uiSmallBufferSize);
    }
    
    return m_pSmallBuffer;
}

__END_NAMESPACE



//=============================================================================
// END OF FILE
//=============================================================================
