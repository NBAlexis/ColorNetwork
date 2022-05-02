//=============================================================================
// FILENAME : Random.h
// 
// DESCRIPTION:
//
//
// REVISION:
//  [12/6/2018 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

__global__ void _CN_LAUNCH_BOUND
_kernalAllocateSeedTable(UINT* pDevicePtr, UINT uiSeed)
{
    UINT uiThread = threadIdx.x;
    CRandom::_deviceAsignSeeds(pDevicePtr, uiSeed + uiThread, uiThread);
}

__global__ void _CN_LAUNCH_BOUND
_kernalInitialXORWOW(curandState * states, UINT uiSeed)
{
    UINT uiThread = threadIdx.x;
    curand_init(uiSeed, uiThread, 0, &states[uiThread]);
}

__global__ void _CN_LAUNCH_BOUND
_kernalInitialPhilox(curandStatePhilox4_32_10_t* states, UINT uiSeed)
{
    UINT uiThread = threadIdx.x;
    curand_init(uiSeed, uiThread, 0, &states[uiThread]);
}

__global__ void _CN_LAUNCH_BOUND
_kernalInitialMRG(curandStateMRG32k3a* states, UINT uiSeed)
{
    UINT uiThread = threadIdx.x;
    curand_init(uiSeed, uiThread, 0, &states[uiThread]);
}

__global__ void _CN_LAUNCH_BOUND
_kernalInitialSobel32(curandStateSobol32* states, curandDirectionVectors32_t* dirs, UINT uiSeed)
{
    UINT uiThread = threadIdx.x;
    curand_init(dirs[uiThread], uiSeed % 16, &states[uiThread]);
}

__global__ void _CN_LAUNCH_BOUND
_kernalInitialScrambledSobel32(curandStateScrambledSobol32* states, UINT* consts, curandDirectionVectors32_t* dirs, UINT uiSeed)
{
    UINT uiThread = threadIdx.x;
    curand_init(dirs[uiThread], consts[uiThread], uiSeed % __SOBEL_OFFSET_MAX, &states[uiThread]);
}

CRandom::~CRandom()
{

    switch (m_eRandomType)
    {
    case ER_Schrage:
        {
            appCudaFree(m_pDeviceSeedTable);
        }
        break;
    case ER_MRG32K3A:
        {
            checkCudaErrors(curandDestroyGenerator(m_HGen));
            checkCudaErrors(cudaFree(m_deviceBuffer));
            appCudaFree(m_pDeviceRandStatesMRG);
        }
        break;
    case ER_PHILOX4_32_10:
        {
            checkCudaErrors(curandDestroyGenerator(m_HGen));
            checkCudaErrors(cudaFree(m_deviceBuffer));
            appCudaFree(m_pDeviceRandStatesPhilox);
        }
        break;
    case ER_QUASI_SOBOL32:
        {
            checkCudaErrors(curandDestroyGenerator(m_HGen));
            checkCudaErrors(cudaFree(m_deviceBuffer));
            appCudaFree(m_pDeviceRandStatesSobol32);
            appCudaFree(m_pDeviceSobolDirVec);
        }
        break;
    case ER_SCRAMBLED_SOBOL32:
        {
            checkCudaErrors(curandDestroyGenerator(m_HGen));
            checkCudaErrors(cudaFree(m_deviceBuffer));
            appCudaFree(m_pDeviceRandStatesScrambledSobol32);
            appCudaFree(m_pDeviceSobolDirVec);
            appCudaFree(m_pDeviceSobelConsts);
        }
        break;
    case ER_XORWOW:
        default:
        {
            checkCudaErrors(curandDestroyGenerator(m_HGen));
            checkCudaErrors(cudaFree(m_deviceBuffer));
            appCudaFree(m_pDeviceRandStatesXORWOW);
        }
        break;
    }
}

//Initial XORWOW only support 512 threads per block
void CRandom::InitialStatesXORWOW()
{
    appCudaMalloc((void **)&m_pDeviceRandStatesXORWOW, sizeof(curandState) * m_uiMaxThread);
    _kernalInitialXORWOW <<<1, m_uiMaxThread >>> (m_pDeviceRandStatesXORWOW, m_uiHostSeed);
}

//Initial Philox only support 256 threads per block
void CRandom::InitialStatesPhilox()
{
    appCudaMalloc((void **)&m_pDeviceRandStatesPhilox, sizeof(curandStatePhilox4_32_10_t) * m_uiMaxThread);
    _kernalInitialPhilox << <1, m_uiMaxThread >> > (m_pDeviceRandStatesPhilox, m_uiHostSeed);
}

//Initial MRG only support 256 threads per block
void CRandom::InitialStatesMRG()
{
    appCudaMalloc((void **)&m_pDeviceRandStatesMRG, sizeof(curandStateMRG32k3a) * m_uiMaxThread);
    _kernalInitialMRG << <1, m_uiMaxThread >> > (m_pDeviceRandStatesMRG, m_uiHostSeed);
}

void CRandom::InitialStatesSobol32()
{
    //support only 20000 dimensions, so using _HC_Volumn instead
    appCudaMalloc((void **)&m_pDeviceRandStatesSobol32, sizeof(curandStateSobol32) * m_uiMaxThread);
    appCudaMalloc((void **)&m_pDeviceSobolDirVec, sizeof(curandDirectionVectors32_t) * m_uiMaxThread);

    //int[32]
    curandDirectionVectors32_t *hostVectors32;
    checkCudaErrors(curandGetDirectionVectors32(&hostVectors32, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
    checkCudaErrors(cudaMemcpy(m_pDeviceSobolDirVec, hostVectors32, sizeof(curandDirectionVectors32_t) * m_uiMaxThread, cudaMemcpyHostToDevice));

    _kernalInitialSobel32 << <1, m_uiMaxThread >> > (m_pDeviceRandStatesSobol32, m_pDeviceSobolDirVec, m_uiHostSeed);
}

void CRandom::InitialStatesScrambledSobol32()
{
    appCudaMalloc((void **)&m_pDeviceRandStatesScrambledSobol32, sizeof(curandStateScrambledSobol32) * m_uiMaxThread);
    appCudaMalloc((void **)&m_pDeviceSobolDirVec, sizeof(curandDirectionVectors32_t) * m_uiMaxThread);
    appCudaMalloc((void **)&m_pDeviceSobelConsts, sizeof(UINT) * m_uiMaxThread);

    curandDirectionVectors32_t *hostVectors32;
    checkCudaErrors(curandGetDirectionVectors32(&hostVectors32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    checkCudaErrors(cudaMemcpy(m_pDeviceSobolDirVec, hostVectors32,  sizeof(curandDirectionVectors32_t) * m_uiMaxThread, cudaMemcpyHostToDevice));

    UINT * hostScrambleConstants32;
    checkCudaErrors(curandGetScrambleConstants32(&hostScrambleConstants32));
    checkCudaErrors(cudaMemcpy(m_pDeviceSobelConsts, hostScrambleConstants32, sizeof(UINT) * m_uiMaxThread, cudaMemcpyHostToDevice));

    _kernalInitialScrambledSobel32 << <1, m_uiMaxThread >> > (m_pDeviceRandStatesScrambledSobol32, m_pDeviceSobelConsts, m_pDeviceSobolDirVec, m_uiHostSeed);
}

void CRandom::InitialTableSchrage()
{
    appCudaMalloc((void **)&m_pDeviceSeedTable, sizeof(UINT) * m_uiMaxThread);
    _kernalAllocateSeedTable << <1, m_uiMaxThread >> > (m_pDeviceSeedTable, m_uiMaxThread);
}

#pragma region Test

__global__ void _CN_LAUNCH_BOUND
_kernelMCPi(UINT* output, UINT lengthyz, UINT lengthz, UINT uiLoop, UINT uithreadCount)
{
    __shared__ UINT sData1[1024];
    __shared__ UINT sData2[1024];
    UINT uiToAdd = 0;
    UINT uiToAdd2 = 0;
    //We have a very large grid, but for a block, it is always smaller (or equval to volumn)
    const UINT fatIndex = threadIdx.x * lengthyz + threadIdx.y * lengthz + threadIdx.z;
    for (UINT i = 0; i < uiLoop; ++i)
    {
        const FLOAT x = _deviceRandomF(fatIndex) * 2.0f - 1.0f;
        const FLOAT y = _deviceRandomF(fatIndex) * 2.0f - 1.0f;
        if (x * x + y * y < 1.0f)
        {
            ++uiToAdd;
        }
        ++uiToAdd2;
    }
    sData1[fatIndex] = uiToAdd;
    sData2[fatIndex] = uiToAdd2;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        UINT all1 = 0;
        UINT all2 = 0;
        for (UINT i = 0; i < uithreadCount; ++i)
        {
            all1 += sData1[i];
            all2 += sData2[i];
        }
        //printf("how many?= %d\n", all1);
        atomicAdd(output, all1);
        atomicAdd(output + 1, all2);
    }
}

__global__ void _CN_LAUNCH_BOUND
_kernelMCE(FLOAT* output, UINT lengthyz, UINT lengthz, UINT uiLoop, UINT uithreadCount)
{
    __shared__ FLOAT sData1[1024];
    __shared__ FLOAT sData2[1024];
    FLOAT fToAdd = 0;
    FLOAT fToAdd2 = 0;
    const UINT fatIndex = threadIdx.x * lengthyz + threadIdx.y * lengthz + threadIdx.z;
    for (UINT i = 0; i < uiLoop; ++i)
    {
        const _SComplex c = _deviceRandomGaussC(fatIndex);
        fToAdd += (c.x + c.y);
        fToAdd2 += (c.x * c.x + c.y * c.y);
    }
    sData1[fatIndex] = fToAdd;
    sData2[fatIndex] = fToAdd2;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        FLOAT all1 = 0;
        FLOAT all2 = 0;
        for (UINT i = 0; i < uithreadCount; ++i)
        {
            all1 += sData1[i];
            all2 += sData2[i];
        }
        //printf("how many?= %d\n", all1);
        atomicAdd(output, all1);
        atomicAdd(output + 1, all2);
    }
}

FLOAT CNAPI CalculatePi(const TArray<UINT> & decompose)
{
    dim3 blocknumber(decompose[0], decompose[1], decompose[2]);
    dim3 threadnumber(decompose[3], decompose[4], decompose[5]);
    const UINT threadCount = decompose[3] * decompose[4] * decompose[5];
    const UINT lengthyz = decompose[4] * decompose[5];
    const UINT lengthz = decompose[5];
    const UINT total = decompose[0] * decompose[1] * decompose[2] * decompose[3] * decompose[4] * decompose[5] * decompose[6];
    const UINT uiLoop = decompose[6];

    UINT outPutHost[2];
    outPutHost[0] = 0;
    outPutHost[1] = 0;

    UINT *outPut;
    checkCudaErrors(cudaMalloc((void**)&outPut, sizeof(UINT) * 2));
    checkCudaErrors(cudaMemcpy(outPut, outPutHost, sizeof(UINT) * 2, cudaMemcpyHostToDevice));

    _kernelMCPi << <blocknumber, threadnumber >> > (outPut, lengthyz, lengthz, uiLoop, threadCount);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(outPutHost, outPut, sizeof(UINT) * 2, cudaMemcpyDeviceToHost));

    appParanoiac(_T("==== results: %d / %d \n"), outPutHost[0], outPutHost[1]);

    return 4.0f * outPutHost[0] / (FLOAT)(total);
}

FLOAT CNAPI CalculateE(const TArray<UINT> & decompose)
{
    dim3 blocknumber(decompose[0], decompose[1], decompose[2]);
    dim3 threadnumber(decompose[3], decompose[4], decompose[5]);
    const UINT threadCount = decompose[3] * decompose[4] * decompose[5];
    const UINT lengthyz = decompose[4] * decompose[5];
    const UINT lengthz = decompose[5];
    const UINT total = decompose[0] * decompose[1] * decompose[2] * decompose[3] * decompose[4] * decompose[5] * decompose[6];
    const UINT uiLoop = decompose[6];

    FLOAT outPutHost[2];
    outPutHost[0] = 0.0F;
    outPutHost[1] = 0.0F;

    FLOAT*outPut;
    checkCudaErrors(cudaMalloc((void**)&outPut, sizeof(FLOAT) * 2));
    checkCudaErrors(cudaMemcpy(outPut, outPutHost, sizeof(FLOAT) * 2, cudaMemcpyHostToDevice));

    _kernelMCE << <blocknumber, threadnumber >> > (outPut, lengthyz, lengthz, uiLoop, threadCount);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(outPutHost, outPut, sizeof(FLOAT) * 2, cudaMemcpyDeviceToHost));

    const FLOAT fAv = outPutHost[0] / (2.0f * total);
    const FLOAT fBv = outPutHost[1] / (2.0f * total) - fAv * fAv;

    return sqrt(fBv);
}

#pragma endregion

__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
