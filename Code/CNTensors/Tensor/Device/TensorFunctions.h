//=============================================================================
// FILENAME : TensorFunctions.h
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [28/05/2020 nbale]
//=============================================================================

#ifndef _TENSORFUNCTIONS_H_
#define _TENSORFUNCTIONS_H_

__BEGIN_NAMESPACE

#pragma region Index mapping

/**
 * A 32 order tensor with all dim=2 needs 64G memory
 * Any tensor larger than this is not capable
 */
constexpr BYTE _kMaxSupportedOrder = 32;

/**
 * In the case of TN, typically we have a tensor smaller than this
 */
constexpr BYTE _kSmallOrder = 8;
constexpr WORD _kSmallMaxDim = 256;
typedef WORD smallIdx;

/**
 * We only use the block.x and thread.x for simplicity.
 * blocks * threads = \prod_i lengths[i], with threads < uiMaxThread
 */
static inline void SimpleThreadDecompose(
    const UINT* __restrict__ lengths, 
    BYTE byIndexCount, 
    UINT& blocks, 
    UINT& threads,
    UINT uiMaxThread = BOUND_THREAD)
{
    blocks = 1;
    threads = 1;
    for (BYTE i = 0; i < byIndexCount; ++i)
    {
        if (lengths[i] * threads < uiMaxThread)
        {
            threads = threads * lengths[i];
        }
        else
        {
            blocks = blocks * lengths[i];
        }
    }
}

/**
 * Map the linear index of thread to linear index of tensor
 * multpliedLengths[i] = \prod lengths[j>i]
 * (1) Calculate (a1, a2, ...) from uiIdx with multpliedLengths
 * (2) Calcuate linear index of T[a1+start1, a2+start2, ...] with strides
 * Take index 1234 for example, multpliedLengths = {1000, 100, 10}
 */
__device__ static __inline__ UINT IndexWithStrideWithStart(
    const UINT* __restrict__ strides,
    const UINT* __restrict__ starts,
    const UINT* __restrict__ multpliedLengths,
    UINT uiIdx, BYTE byOrder)
{
    UINT uiRet = 0;
    for (UINT i = 0; i < byOrder; ++i)
    {
        /**
         * Take 1234 for example
         * when i = 0,      1 = 1234 / 1000
         * when i = 1,2,    2 = (1234 % 1000) / 100,  3 = (1234 % 100) / 10
         * when i = 3,      4 = 1234 % 100
         */
        const UINT idxForThisOrder = (0 == i)
            ? (uiIdx / multpliedLengths[i])
            : ((byOrder == i + 1)
                ? (uiIdx % multpliedLengths[i - 1])
                : ((uiIdx % multpliedLengths[i - 1]) / multpliedLengths[i]));

        uiRet = uiRet + (starts[i] + idxForThisOrder) * strides[i];
    }
    return uiRet;
}

/**
 * Similar as indexWithStrideWithStart, but no start
 */
__device__ static __inline__ UINT IndexWithStride(
    const UINT* __restrict__ strides,
    const UINT* __restrict__ multpliedLengths,
    UINT uiIdx, BYTE byOrder)
{
    UINT uiRet = 0;
    for (UINT i = 0; i < byOrder; ++i)
    {
        const UINT idxForThisOrder = (0 == i)
            ? (uiIdx / multpliedLengths[i])
            : ((byOrder == i + 1) 
                ? (uiIdx % multpliedLengths[i - 1])
                : ((uiIdx % multpliedLengths[i - 1]) / multpliedLengths[i]));

        uiRet = uiRet + idxForThisOrder * strides[i];
    }
    return uiRet;
}

/**
 * a 128-bit index
 * I think use array is better than bit as long as it is properly aligned
 */
struct SWorkingIndex
{
    smallIdx m_Idx[_kSmallOrder];
};

/**
 * map thread index to working index
 */
__device__ static __inline__ SWorkingIndex ThreadIndexToWorkIndex(UINT uiIdx,
    const UINT* __restrict__ multpliedLengths, BYTE byOrder)
{
    SWorkingIndex ret;

    //#pragma unroll
    //for (BYTE i = 0; i < _kSmallOrder; ++i)
    //{
    //    if (i < byOrder)
    //    {
    //        ret.m_Idx[i] = static_cast<smallIdx>((0 == i) ? (uiIdx / multpliedLengths[i])
    //            : ((byOrder == i + 1)
    //                ? (uiIdx % multpliedLengths[i - 1])
    //                : ((uiIdx % multpliedLengths[i - 1]) / multpliedLengths[i])));
    //    }
    //    else
    //    {
    //        ret.m_Idx[i] = 0;
    //    }
    //}
    //unroll in .h file is unknown pragma
    
    #define ___addidx1(i) \
    if (i < byOrder) \
    { \
        ret.m_Idx[i] = static_cast<smallIdx>((0 == i) ? (uiIdx / multpliedLengths[i]) \
            : ((byOrder == i + 1) \
                ? (uiIdx % multpliedLengths[i - 1]) \
                : ((uiIdx % multpliedLengths[i - 1]) / multpliedLengths[i]))); \
    } \
    else \
    { \
        ret.m_Idx[i] = 0; \
    }

    ___addidx1(0);
    ___addidx1(1);
    ___addidx1(2);
    ___addidx1(3);
    ___addidx1(4);
    ___addidx1(5);
    ___addidx1(6);
    ___addidx1(7);

    return ret;
}

__device__ static __inline__ UINT WorkIndexToTensorIndexWithStart(
    const smallIdx* __restrict__ wkidx, 
    const UINT* __restrict__ strides,
    const UINT* __restrict__ starts, 
    BYTE byOrder)
{
    UINT uiRet = 0;
    //#pragma unroll
    //for (BYTE i = 0; i < _kSmallOrder; ++i)
    //{
    //    if (i < byOrder)
    //    {
    //        uiRet = uiRet + (wkidx[i] + starts[i]) * strides[i];
    //    }
    //}
    //unroll in .h file is unknown pragma

    #define ___addidx2(i) \
    if (i < byOrder) \
    { \
        uiRet = uiRet + (wkidx[i] + starts[i]) * strides[i]; \
    }

    ___addidx2(0);
    ___addidx2(1);
    ___addidx2(2);
    ___addidx2(3);
    ___addidx2(4);
    ___addidx2(5);
    ___addidx2(6);
    ___addidx2(7);

    return uiRet;
}

__device__ static __inline__ UINT WorkIndexToTensorIndex(
    const smallIdx* __restrict__ wkidx,
    const UINT* __restrict__ strides,
    BYTE byOrder)
{
    UINT uiRet = 0;
    //#pragma unroll
    //for (BYTE i = 0; i < _kSmallOrder; ++i)
    //{
    //    if (i < byOrder)
    //    {
    //        uiRet = uiRet + wkidx[i] * strides[i];
    //    }
    //}
    //unroll in .h file is unknown pragma
    #define ___addidx3(i) \
    if (i < byOrder) \
    { \
        uiRet = uiRet + wkidx[i] * strides[i]; \
    }

    ___addidx3(0);
    ___addidx3(1);
    ___addidx3(2);
    ___addidx3(3);
    ___addidx3(4);
    ___addidx3(5);
    ___addidx3(6);
    ___addidx3(7);

    return uiRet;
}


#pragma endregion

#pragma region Working Space

class CNAPI CTensorOpWorkingSpace
{
public:
    enum
    {
        _kSmallBufferSize = 65536,
        _kAlignByteMinusOne = 7,
    };

    CTensorOpWorkingSpace();
    ~CTensorOpWorkingSpace();

    /**
     * Note: Every call to GetSmallDeviceBuffer will make previous buffer unsafe
     */
    BYTE* GetSmallDeviceBuffer(UINT uiLength);

    const UINT* GetZeroStartBuffer() const { return m_pDeviceZeroStart; }
    UINT* GetMultiplyLengthBuffer() { return m_pMultiplyLengthBuffer; }

protected:

    BYTE* m_pSmallBuffer;
    UINT m_uiSmallBufferIdx;
    UINT* m_pDeviceZeroStart;
    UINT m_pMultiplyLengthBuffer[_kMaxSupportedOrder];
};

CTensorOpWorkingSpace* appGetTensorOpWorkingSpace();

#pragma endregion

#pragma region Common

/**
 * if mask[...] != 0, then dst[..., i0 + i, ..., j0 + j] = src[..., i0' + i, ..., j0' + j]
 */
template <class T> CNAPI void BlockCopyMasked(
    T* dst,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ dstIndexStart,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount);

/**
 * dst[..., i0 + i, ..., j0 + j] = src[..., i0' + i, ..., j0' + j]
 * When dstIndexStart or srcIndexStart are 0, treat as start from 0
 */
template <class T> inline void BlockCopy(
    T* dst,
    const UINT* __restrict__ dstStride,
    const UINT* __restrict__ dstIndexStart,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    BlockCopyMasked(dst, dstStride, dstIndexStart, src, srcStride, srcIndexStart, NULL, NULL, NULL, lengths, byIndexCount);
}

/**
 * dst[..., i, ..., j] = src[..., i, ..., j]
 * When dstIndexStart or srcIndexStart are 0, treat as start from 0
 */
template <class T> inline void BlockCopy(
    T* dst,
    const UINT* __restrict__ dstStride,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    BlockCopy(dst, dstStride, NULL, src, srcStride, NULL, lengths, byIndexCount);
}

/**
 * if mask[...] != 0, then T[..., i0 + i, ..., j0 + j] = val
 */
template <class T> __DLL_EXPORT void FillMasked(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcIndexStart,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount);

/**
 * T[..., i0 + i, ..., j0 + j] = val
 */
template <class T> inline void Fill(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    FillMasked(src, val, srcStride, srcStart, NULL, NULL, NULL, lengths, byIndexCount);
}

/**
 * T[..., i, ..., j] = val
 */
template <class T> inline void Fill(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ lengths, BYTE byIndexCount)
{
    Fill(src, val, srcStride, NULL, lengths, byIndexCount);
}

template <class T> __DLL_EXPORT void Random(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount);


template <class T> __DLL_EXPORT void RandomMasked(T* src, const T& val,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcStart,
    const UINT* __restrict__ srcEnd,
    const BYTE* __restrict__ mask,
    const UINT* __restrict__ maskStride,
    const UINT* __restrict__ maskIndexStart,
    const UINT* __restrict__ lengths, BYTE byIndexCount);

/**
 * T[x1, x2, x3, ..., y1, y2, y3, ...] = T[ x2, y1, x1, x3, y4, ...]
 * order Stride and Dims as stride(dim) of x1, x2, x3, ..., y1, y2, y3.
 * Note: transpose must copy
 */
template <class T> __DLL_EXPORT void Transpose(T* dst,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcDim,
    BYTE byIndexCount);

/**
 * M[x, y] = T[ x2, y1, x1, x3, y4, ...]
 * order Stride and Dims as stride(dim) of x1, x2, x3, ..., y1, y2, y3, Then everything automatically happen
 *
 * Explanation:
 * threadIdx = x * dimy + y
 * x1 = threadIdx / x2*x3*...*y1*y2*y3
 * x2 = (threadIdx % x2*x3*...*y1*y2*y3) % (x3*...*y1*y2*y3)
 * so the first thing is to make sure multiply lengthes are ordered.
 * TensorIdx = x1 * stride1 + x2 * stride2 + ...
 * so the next thing is to make sure the strides are ordered.
 */
template <class T> inline void ToMatrix(T* dst,
    const T* __restrict__ src,
    const UINT* __restrict__ srcStride,
    const UINT* __restrict__ srcDim,
    BYTE byIndexCount)
{
    Transpose(dst, src, srcStride, srcDim, byIndexCount);
}

/**
 * Print as a vector
 */
template <class T> __DLL_EXPORT void DebugPrint(
    const T* __restrict__ src,
    UINT uiSize);

/**
 * Print as a matrix
 */
template <class T> __DLL_EXPORT void DebugPrint(
    const T* __restrict__ src,
    UINT uiXDim,
    UINT uiYDim);

#pragma endregion

#if 0

template <class T> CNAPI T Dot(T* self, T* src);

template <class T> CNAPI void Add(T* self, T* t, T* mat, T* vec, T beta, T alpha);
template <class T> CNAPI void MV(T* self, T* t, T* mat, T* vec, T beta, T alpha);
template <class T> CNAPI void AddMV(T* self, T* t, T* mat, T* vec, T beta, T alpha);

template <class T> CNAPI void MM(T* self, T* t, T* mat1, T* mat2, T beta, T alpha);
template <class T> CNAPI void AddMM(T* self, T* t, T* mat1, T* mat2, T beta, T alpha);
template <class T> CNAPI void AddR(T* self, T* t, T* vec1, T* vec2, T beta, T alpha);
template <class T> CNAPI void AddBMM(T* result, T* t, T* batch1, T* batch2, T beta, T alpha);
template <class T> CNAPI void BAddBMM(T* result, T* t, T* batch1, T* batch2, T beta, T alpha);

#endif

__END_NAMESPACE

#endif //#ifndef _TENSORFUNCTIONS_H_


//=============================================================================
// END OF FILE
//=============================================================================
