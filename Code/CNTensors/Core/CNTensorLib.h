//=============================================================================
// FILENAME : CNTensorLib.h
// 
// DESCRIPTION:
//
// REVISION:
//  [31/05/2020 nbale]
//=============================================================================

#ifndef _CNTENSORLIB_H_
#define _CNTENSORLIB_H_

__BEGIN_NAMESPACE

class CNAPI CNTensorLib
{
public:
    CNTensorLib();
    ~CNTensorLib();

    /**
     * Make sure call Initial after GTrace is initialed
     */
    void Initial(const TCHAR* sConfigFile);

    /**
     * Make sure it always safe to call Initial again after Exit
     */
    void Exit();

    class CCudaHelper* GetCuda() const { return m_pCuda; }
    class CTensorOpWorkingSpace* GetOpWorkingSpace() const { return m_pOpWorkingSpace; }
    class CRandom* GetRandom() const { return m_pRandom; }

protected:

    class CCudaHelper* m_pCuda;
    class CTensorOpWorkingSpace* m_pOpWorkingSpace;
    class CRandom* m_pRandom;
    class CRandom* m_pDeviceRandom;

    void InitialRandom();
};

#pragma region Globle functions

extern CNAPI CNTensorLib GCNLib;

inline void appInitialCNLib(const TCHAR* sConfigFile) { GCNLib.Initial(sConfigFile); }
inline void appExitCNLib() { GCNLib.Exit(); }

inline class CCudaHelper* appGetCuda() { return GCNLib.GetCuda(); }
inline class CRandom* appGetRandom() { return GCNLib.GetRandom(); }
inline class CTensorOpWorkingSpace* appGetOpWorkingSpace() { return GCNLib.GetOpWorkingSpace(); }

#pragma endregion

__END_NAMESPACE

#endif //#ifndef _CNTENSORLIB_H_

//=============================================================================
// END OF FILE
//=============================================================================