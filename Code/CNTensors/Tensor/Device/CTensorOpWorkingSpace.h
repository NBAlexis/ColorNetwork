//=============================================================================
// FILENAME : CTensorOpWorkingSpace.h
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [21/01/2022 nbale]
//=============================================================================

#ifndef _CTENSOROPWORKINGSPACE_H_
#define _CTENSOROPWORKINGSPACE_H_

__BEGIN_NAMESPACE

/**
 * A 32 order tensor with all dim=2 needs 64G memory
 * Any tensor larger than this is not capable
 * MOVE THIS DEFINITION TO SOME CONSTANT.H FILE
 */
constexpr BYTE _kMaxSupportedOrder = 32;

class CNAPI CTensorOpWorkingSpace
{
public:
    enum
    {
        _kSmallBufferSize = 4095,
        _kAlignByteMinusOne = 7,
    };

    CTensorOpWorkingSpace();
    ~CTensorOpWorkingSpace();

    /**
     * Note: Every call to GetSmallDeviceBuffer will make previous buffer unsafe
     */
    BYTE* GetSmallDeviceBuffer(UINT uiSize);

    const UINT* GetZeroStartBuffer() const { return m_pDeviceZeroStart; }
    UINT* GetMultiplyLengthBuffer() { return m_pMultiplyLengthBuffer; }

protected:

    BYTE* m_pSmallBuffer;
    UINT m_uiSmallBufferSize;
    UINT* m_pDeviceZeroStart;
    UINT m_pMultiplyLengthBuffer[_kMaxSupportedOrder];
};

inline BYTE* appGetSmallDeviceBuffer(UINT uiSize)
{
    return appGetOpWorkingSpace()->GetSmallDeviceBuffer(uiSize);
}

__END_NAMESPACE

#endif //#ifndef _CTENSOROPWORKINGSPACE_H_


//=============================================================================
// END OF FILE
//=============================================================================
