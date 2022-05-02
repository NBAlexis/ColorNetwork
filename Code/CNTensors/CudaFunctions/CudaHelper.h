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
constexpr UINT kSmallBufferSize = 65536;

extern __constant__ INT _constIntegers[kContentLength];
//extern __constant__ Real _constFloats[kContentLength];

/**
* Note that, the pointers are copyied here. So the virtual functions should not be used!
*/
extern __constant__ class CRandom* __r;

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

struct CudaMemInfo
{
    UINT m_uiReason;
    UINT m_uiSize;
    PTRINT m_pPointer;
    TCHAR m_sInfo[256];
};

class CNAPI CCudaHelper
{
public:

    CCudaHelper()
        : m_pSmallSizeBuffer(NULL)
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
    void MemoryQuery();

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

    /**
     * For using those in .h files
     * malloc and free is not static constant for
     * furture possible for statistics for memory usage
     */
    void _Malloc(const TCHAR* sLocation, void** ptr, UINT uiSize, UINT uiReason = 0);
    void Free(void* ptr);

    /**
     * (1) To avoid checkCudaErrors in .h file
     * (2) Furture possible for statistics for memory copy
     */
    void CopyDD(void* dest, const void* src, UINT uiSize);
    void CopyHD(void* dest, const void* src, UINT uiSize);
    void CopyDH(void* dest, const void* src, UINT uiSize);

    BYTE* GetSmallBuffer() const
    {
        return m_pSmallSizeBuffer;
    }

protected:

    /**
     * This is a buffer for one time usage
     */
    BYTE* m_pSmallSizeBuffer;
    THashMap<PTRINT, CudaMemInfo> m_pMemRecord;

    #pragma endregion
};

inline class CCudaHelper* appGetCuda();

extern CNAPI void _appCudaFree(void* ptr);

__END_NAMESPACE

#define appCudaFree(ptr) _appCudaFree(ptr); ptr=NULL
#define appCudaMalloc(...) {TCHAR ___msg[1024];appSprintf(___msg, 1024, _T("%s(%d)"), _T(__FILE__), __LINE__); appGetCuda()->_Malloc(___msg, __VA_ARGS__);}

#endif //#ifndef _CUDAHELPER_H_

//=============================================================================
// END OF FILE
//=============================================================================