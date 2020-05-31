//=============================================================================
// FILENAME : CLGFFT.h
// 
// DESCRIPTION:
// This is helper to calculate FFT using cufft
//
//
// REVISION:
//  [09/16/2019 nbale]
//=============================================================================

#if 0
//#ifndef _FFT_H_
#define _FFT_H_

__BEGIN_NAMESPACE

/**
 * We prefer ES_None as default, because we will have to multiply other quantities
 */
enum EFFT_Scale
{
    ES_None,
    ES_1OverNForward,
    ES_1OverNInverse,
    ES_1OverSqrtNBoth,
};

class CLGAPI CCLGFFTHelper
{
public:

    CCLGFFTHelper()
    : m_pDeviceBuffer(NULL)
    {
        
    }

    ~CCLGFFTHelper()
    {
        if (NULL != m_pDeviceBuffer)
        {
            checkCudaErrors(cudaFree(m_pDeviceBuffer));
        }
    }

    /**
     * copied is the copy of the source, will be changed
     */
    static UBOOL FFT3DWithXYZ(CNComplex* copied, TArray<INT> dims, UBOOL bForward);
    static UBOOL FFT3DWithXYZW(CNComplex* copied, TArray<INT> dims, UBOOL bForward);
    static UBOOL FFT4DWithXYZW(CNComplex* copied, TArray<INT> dims, UBOOL bForward);
    static UBOOL FFT3D(CNComplex* res, UBOOL bForward, EFFT_Scale eScale = ES_None);
    static UBOOL FFT4D(CNComplex* res, UBOOL bForward, EFFT_Scale eScale = ES_None);

    UBOOL FFT3DSU3(deviceSU3* res, UBOOL bForward, EFFT_Scale eScale = ES_None);
    UBOOL FFT4DSU3(deviceSU3* res, UBOOL bForward, EFFT_Scale eScale = ES_None);

    /**
     * Test function
     */
    static void TestFFT();

private:

    static void GenerateTestArray(CNComplex* hostArray, INT iSize);
    static void PrintTestArray3D(CNComplex* hostArray);
    static void PrintTestArray4D(CNComplex* hostArray);
    void CheckBuffer();

    CNComplex* m_pDeviceBuffer;
};

__END_NAMESPACE

#endif //#ifndef _FFT_H_

//=============================================================================
// END OF FILE
//=============================================================================
