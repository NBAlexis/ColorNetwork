//=============================================================================
// FILENAME : COpenCLStart.h
// 
// DESCRIPTION:
//
// REVISION:
//  [31/08/2021 nbale]
//=============================================================================

#ifndef _OPENCLSTART_H_
#define _OPENCLSTART_H_

#if _CN_GLOBALE_AS_CONST
#define _DEVICE_CONST const __global
#else
#define _DEVICE_CONST __constant
#endif

__BEGIN_NAMESPACE

class CNAPI COpenCLStart
{
public:
    COpenCLStart();
    ~COpenCLStart();

    /**
     * Make sure call Initial after GTrace is initialed
     */
    void Initial(const class CParameters &sConfigFile);

    /**
     * Make sure it always safe to call Initial again after Exit
     */
    void Exit();

    void PrintDeviceInfo();

protected:

    

};

__END_NAMESPACE

#endif //#ifndef _OPENCLSTART_H_

//=============================================================================
// END OF FILE
//=============================================================================