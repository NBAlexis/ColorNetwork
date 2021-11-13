//=============================================================================
// FILENAME : TensorFunctions_WorkingSpace.cu
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
    , m_uiSmallBufferIdx(0)
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

BYTE* CTensorOpWorkingSpace::GetSmallDeviceBuffer(UINT uiLength)
{
    const UINT alignedLength = (uiLength + _kAlignByteMinusOne) & (~_kAlignByteMinusOne);
    const UINT uiOldIndex = m_uiSmallBufferIdx;
    m_uiSmallBufferIdx = m_uiSmallBufferIdx + alignedLength;
    if (m_uiSmallBufferIdx < _kSmallBufferSize)
    {
        return m_pSmallBuffer + uiOldIndex;
    }

    appCudaFree(m_pSmallBuffer);
    appCudaMalloc((void**)&m_pSmallBuffer, _kSmallBufferSize);
    m_uiSmallBufferIdx = alignedLength;
    return m_pSmallBuffer;
}

CTensorOpWorkingSpace* appGetTensorOpWorkingSpace()
{
    return NULL;
}

__END_NAMESPACE



//=============================================================================
// END OF FILE
//=============================================================================
