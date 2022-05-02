//=============================================================================
// FILENAME : Random.h
// 
// DESCRIPTION:
// This is random number for parallel
//
//
// REVISION:
//  [18/04/2022 nbale]
//=============================================================================

#ifndef _RANDOM_H_
#define _RANDOM_H_


#define __SOBEL_OFFSET_MAX (4096)

#define __DefineRandomFuncion(rettype, funcname) __device__ __inline__ static rettype _deviceRandom##funcname(UINT uiThreadIdx) \
{ \
    return __r->_deviceRandom##funcname(uiThreadIdx); \
}


__BEGIN_NAMESPACE

__DEFINE_ENUM (ERandom,
    ER_Schrage,

    ER_XORWOW,
    ER_MRG32K3A,
    //ER_MTGP32, //see the document, this has lots of constraints.
    ER_PHILOX4_32_10,
    ER_QUASI_SOBOL32,
    ER_SCRAMBLED_SOBOL32,

    ER_ForceDWORD = 0x7fffffff,
    )

__DEFINE_ENUM (ERandomSeedType,
    ERST_Number,
    ERST_Timestamp,

    ERST_ForceDWORD = 0x7fffffff,
    )

class CNAPI CRandom
{
public:

    /**
    * There are nine types of random number generators in cuRAND, that fall into two categories. 
    *  CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_PHILOX4_32_10 and CURAND_RNG_PSEUDO_MT19937 are pseudorandom number generators. 
    * CURAND_RNG_PSEUDO_XORWOW is implemented using the XORWOW algorithm, a member of the xor-shift family of pseudorandom number generators. 
    * CURAND_RNG_PSEUDO_MRG32K3A is a member of the Combined Multiple Recursive family of pseudorandom number generators. 
    *  CURAND_RNG_PSEUDO_MT19937 and CURAND_RNG_PSEUDO_MTGP32 are members of the Mersenne Twister family of pseudorandom number generators. 
    * CURAND_RNG_PSEUDO_MTGP32 has parameters customized for operation on the GPU. 
    * CURAND_RNG_PSEUDO_MT19937 has the same parameters as CPU version, but ordering is different. 
    * CURNAD_RNG_PSEUDO_MT19937 supports only HOST API and can be used only on architecture sm_35 or higher. 
    * CURAND_RNG_PHILOX4_32_10 is a member of Philox family, which is one of the three non-cryptographic Counter Based Random Number Generators presented on SC11 conference by D E Shaw Research. 
    *
    *  There are 4 variants of the basic SOBOL¡¯ quasi random number generator. All of the variants generate sequences in up to 20,000 dimensions. CURAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SOBOL64, and CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 are quasirandom number generator types. 
    * CURAND_RNG_QUASI_SOBOL32 is a Sobol¡¯ generator of 32-bit sequences. 
    * CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 is a scrambled Sobol¡¯ generator of 32-bit sequences. 
    * CURAND_RNG_QUASI_SOBOL64 is a Sobol¡¯ generator of 64-bit sequences. 
    * CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 is a scrambled Sobol¡¯ generator of 64-bit sequences.
    */
    CRandom(UINT uiSeed, UINT uiMaxThread, ERandom er) 
        : m_eRandomType(er)
        , m_uiMaxThread(uiMaxThread)
        , m_uiHostSeed(uiSeed)
    { 
        switch (er)
        {
            case ER_Schrage:
                {
                    InitialTableSchrage();
                }
                break;
            case ER_MRG32K3A:
                {
                    checkCudaErrors(curandCreateGenerator(&m_HGen, CURAND_RNG_PSEUDO_MRG32K3A));
                    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(m_HGen, uiSeed));
                    checkCudaErrors(cudaMalloc((void**)&m_deviceBuffer, sizeof(FLOAT)));
                    InitialStatesMRG();
                }
                break;
            case ER_PHILOX4_32_10:
                {
                    checkCudaErrors(curandCreateGenerator(&m_HGen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
                    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(m_HGen, uiSeed));
                    checkCudaErrors(cudaMalloc((void**)&m_deviceBuffer, sizeof(FLOAT)));
                    InitialStatesPhilox();
                }
                break;
            case ER_QUASI_SOBOL32:
                {
                    //for sobol, on the host, we use XORWOW
                    checkCudaErrors(curandCreateGenerator(&m_HGen, CURAND_RNG_QUASI_SOBOL32));
                    checkCudaErrors(cudaMalloc((void**)&m_deviceBuffer, sizeof(FLOAT)));
                    InitialStatesSobol32();
                }
                break;
            case ER_SCRAMBLED_SOBOL32:
                {
                    //for sobol, on the host, we use XORWOW
                    checkCudaErrors(curandCreateGenerator(&m_HGen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
                    checkCudaErrors(cudaMalloc((void**)&m_deviceBuffer, sizeof(FLOAT)));
                    InitialStatesScrambledSobol32();
                }
                break;
            case ER_XORWOW:
            default:
                {
                    checkCudaErrors(curandCreateGenerator(&m_HGen, CURAND_RNG_PSEUDO_XORWOW));
                    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(m_HGen, uiSeed));
                    checkCudaErrors(cudaMalloc((void**)&m_deviceBuffer, sizeof(FLOAT)));
                    InitialStatesXORWOW();
                }
                break;
        }

        checkCudaErrors(cudaGetLastError());
    }

    ~CRandom();

    /**
    * Note that this gives [0, 1), and curand_uniform gives (0, 1]
    */
    __device__ __inline__ FLOAT _deviceRandomF(UINT uithreadIdx) const
    {
        switch (m_eRandomType)
        {
            case ER_Schrage:
                return static_cast<FLOAT>(AM * _deviceRandomUISchrage(uithreadIdx));
            case ER_MRG32K3A:
                return 1.0f - curand_uniform(&(m_pDeviceRandStatesMRG[uithreadIdx]));
            case ER_PHILOX4_32_10:
                return 1.0f - curand_uniform(&(m_pDeviceRandStatesPhilox[uithreadIdx]));
            case ER_QUASI_SOBOL32:
                return 1.0f - curand_uniform(&(m_pDeviceRandStatesSobol32[(uithreadIdx)]));
            case ER_SCRAMBLED_SOBOL32:
                return 1.0f - curand_uniform(&(m_pDeviceRandStatesScrambledSobol32[uithreadIdx]));
            case ER_XORWOW:
            default:
                return 1.0f - curand_uniform(&(m_pDeviceRandStatesXORWOW[uithreadIdx]));
        }

        //return 0;
    }

    __device__ __inline__ _SComplex _deviceRandomC(UINT uithreadIdx) const
    {
        const FLOAT f1 = _deviceRandomF(uithreadIdx) * 2.0f - 1.0f;
        const FLOAT f2 = _deviceRandomF(uithreadIdx) * 2.0f - 1.0f;
        return make_cuComplex(f1, f2);
    }

    /**
    * The standard deviation of it is 1
    */
    __device__ __inline__ FLOAT _deviceRandomGaussF(UINT uithreadIdx) const
    {
        const FLOAT f1 = _deviceRandomF(uithreadIdx);
        const FLOAT f2 = _deviceRandomF(uithreadIdx) * PI2F;

        const FLOAT oneMinusf1 = 1.0f - f1;
        const FLOAT inSqrt = -2.0f * log(oneMinusf1 > 0.0f ? oneMinusf1 : _CN_FLT_MIN_);
        const FLOAT amplitude = (inSqrt > 0.0f ? sqrt(inSqrt) : 0.0f) * InvSqrt2F;
        return cos(f2) * amplitude;
    }

    __device__ __inline__ _SComplex _deviceRandomGaussC(UINT uithreadIdx) const
    {
        const FLOAT f1 = _deviceRandomF(uithreadIdx);
        const FLOAT f2 = _deviceRandomF(uithreadIdx) * PI2F;
        printf("%f\n", f1);
        const FLOAT oneMinusf1 = 1.0f - f1;
        const FLOAT inSqrt = -2.0f * log(oneMinusf1 > 0.0f ? oneMinusf1 : _CN_FLT_MIN_);
        const FLOAT amplitude = (inSqrt > 0.0f ? sqrt(inSqrt) : 0.0f) * InvSqrt2F;
        return make_cuComplex(cos(f2) * amplitude, sin(f2) * amplitude);
    }

    __device__ __inline__ UINT _deviceRandomI(UINT uithreadIdx, UINT uiMax) const
    {
        return static_cast<UINT>(uiMax * _deviceRandomF(uithreadIdx));
    }

    __device__ __inline__ FLOAT _deviceRandomIF(UINT uithreadIdx, UINT uiMax) const
    {
        return static_cast<FLOAT>(static_cast<UINT>(uiMax * _deviceRandomF(uithreadIdx)));
    }

    __device__ __inline__ _SComplex _deviceRandomZN(UINT uithreadIdx, UINT uiMax) const
    {
        const FLOAT byRandom = static_cast<FLOAT>(_deviceRandomI(uithreadIdx, uiMax));
        const FLOAT arg = PI2F * byRandom / static_cast<FLOAT>(uiMax);
        return make_cuComplex(cos(arg), sin(arg));
    }

    __host__ __inline__ FLOAT GetRandomF()
    {
        if (ER_Schrage == m_eRandomType)
        {
            return static_cast<FLOAT>(AM * GetRandomUISchrage());
        }

        curandGenerateUniform(m_HGen, m_deviceBuffer, 1);
        checkCudaErrors(cudaMemcpy(m_hostBuffer, m_deviceBuffer, sizeof(FLOAT), cudaMemcpyDeviceToHost));
        return static_cast<FLOAT>(m_hostBuffer[0]);
    }

    FLOAT* m_deviceBuffer;
    FLOAT m_hostBuffer[1];
    curandGenerator_t m_HGen;
    ERandom m_eRandomType;
    UINT m_uiMaxThread;

protected:

    void InitialStatesXORWOW();
    void InitialStatesPhilox();
    void InitialStatesMRG();
    void InitialStatesSobol32();
    void InitialStatesScrambledSobol32();

    curandState* m_pDeviceRandStatesXORWOW;
    curandStatePhilox4_32_10_t* m_pDeviceRandStatesPhilox;
    curandStateMRG32k3a* m_pDeviceRandStatesMRG;

    curandStateSobol32* m_pDeviceRandStatesSobol32;
    curandDirectionVectors32_t* m_pDeviceSobolDirVec;
    UINT* m_pDeviceSobelConsts;
    curandStateScrambledSobol32* m_pDeviceRandStatesScrambledSobol32;

#pragma region Schrage

public:

    UINT* m_pDeviceSeedTable;

    /**
    * run on device, parally set the table
    */
    __device__ __inline__ static void _deviceAsignSeeds(UINT* devicePtr, UINT uiSeed, UINT uiFatIndex)
    {
        devicePtr[uiFatIndex] = (1664525UL * (uiFatIndex + uiSeed) + 1013904223UL) & 0xffffffff;
    }

protected:

    void InitialTableSchrage();

    __device__ __inline__ UINT _deviceRandomUISchrage(UINT threadIndex) const
    {
        m_pDeviceSeedTable[threadIndex] = ((1664525UL * m_pDeviceSeedTable[threadIndex] + 1013904223UL) & 0xffffffff);
        return m_pDeviceSeedTable[threadIndex];
    }

    __host__ __inline__ UINT GetRandomUISchrage()
    {
        m_uiHostSeed = (1664525UL * m_uiHostSeed + 1013904223UL) & 0xffffffff;
        return m_uiHostSeed;
    }

    UINT m_uiHostSeed;

#pragma endregion

};

__DefineRandomFuncion(FLOAT, F)

__DefineRandomFuncion(FLOAT, GaussF)

__DefineRandomFuncion(_SComplex, C)

__DefineRandomFuncion(_SComplex, GaussC)

__device__ __inline__ static FLOAT _deviceRandomIF(UINT uiThreadIdx, UINT uiN)
{
    return __r->_deviceRandomIF(uiThreadIdx, uiN);
}

__device__ __inline__ static _SComplex _deviceRandomZN(UINT uiThreadIdx, UINT uiN)
{
    return __r->_deviceRandomZN(uiThreadIdx, uiN);
}


//__DefineRandomFuncion(CNComplex, Z4)
//
//extern CNAPI Real GetRandomReal();

//==========================
//functions for test
//extern Real CNAPI CalculatePi(const TArray<UINT> & decompose);
//
//extern Real CNAPI CalculateE(const TArray<UINT> & decompose);

__END_NAMESPACE

#endif //#ifndef _RANDOM_H_

//=============================================================================
// END OF FILE
//=============================================================================
