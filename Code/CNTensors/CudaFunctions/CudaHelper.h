//=============================================================================
// FILENAME : CudaHelper.h
// 
// DESCRIPTION:
// This is the file for some common CUDA usage
//
// REVISION:
//  [31/05/2020 nbale]
//=============================================================================

#ifndef _CUDAHELPER_H_
#define _CUDAHELPER_H_

__BEGIN_NAMESPACE

constexpr UINT kContentLength = 256;

extern __constant__ INT _constIntegers[kContentLength];
//extern __constant__ Real _constFloats[kContentLength];

/**
* Note that, the pointers are copyied here. So the virtual functions should not be used!
*/
//extern __constant__ class CRandom* __r;

enum EConstIntId
{
    ECI_None,
    ECI_ForceDWORD = 0x7fffffff,
};

enum EConstFloatId
{
    ECF_None,
    ECF_ForceDWORD = 0x7fffffff,
};

class CNAPI CCudaHelper
{
public:

    CCudaHelper()
        /*: m_pComplexBufferThreadCount(NULL)
        , m_pRealBufferThreadCount(NULL)
        , m_pIntBufferThreadCount(NULL)

        , m_pTensorWorkingSpace(NULL)
        */
    {
        //memset(m_ConstIntegers, 0, sizeof(INT) * kContentLength);
        //memset(m_ConstFloats, 0, sizeof(Real) * kContentLength);

        AllocateTemeraryBuffers();
        InitialHelpers();
    }
    ~CCudaHelper();

    #pragma region System infos

public:

    static void DeviceQuery();
    static void MemoryQuery();

    /**ret[0] = max thread count, ret[1,2,3] = max thread for x,y,z per block*/
    static TArray<UINT> GetMaxThreadCountAndThreadPerblock();

    #pragma endregion

    /**
     * This function is for whennever you want to debug some cuda code
     */
    static void DebugFunction();

    #pragma region Fast sum and fast produce

public:

    static inline UINT GetReduceDim(UINT uiLength)
    {
        UINT iRet = 0;
        while ((1U << iRet) < uiLength)
        {
            ++iRet;
        }
        return iRet;
    }

    template<class T> static T ReduceSum(T* deviceBuffer, UINT uiLength = MAX_THREAD);
    template<class T> static T ReduceProd(T* deviceBuffer, UINT uiLength = MAX_THREAD);
    template<class T> static void ThreadBufferInitial(T* deviceBuffer, const T& val, UINT uiLength = MAX_THREAD);


protected:

    /**
    * Call to malloc buffers with same size as thread count
    */
    void AllocateTemeraryBuffers();

    void ReleaseTemeraryBuffers();

    #pragma endregion

    #pragma region Manager constant pointers

public:

    void CopyConstants() const;
    void CopyRandomPointer(const class CRandom* r) const;

protected:

    INT m_ConstIntegers[kContentLength];
    //Real m_ConstFloats[kContentLength];

    #pragma endregion

    #pragma region Other Helper Members

public:

    void InitialHelpers();
    void ReleaseHelpers();

    class CTensorOpWorkingSpace* GetTensorOpWorkingSpace() const { return m_pTensorWorkingSpace; }

protected:

    class CTensorOpWorkingSpace* m_pTensorWorkingSpace;

    #pragma endregion
};

__END_NAMESPACE


#endif //#ifndef _CUDAHELPER_H_

//=============================================================================
// END OF FILE
//=============================================================================