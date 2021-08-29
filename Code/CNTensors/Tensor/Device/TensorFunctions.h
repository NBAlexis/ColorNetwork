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

#pragma region Contract

template <class Tresult, class TLeft, class TRight> __DLL_EXPORT
void MM(Tresult* dest, const TLeft* __restrict__ left, const TRight* __restrict__ right,
    BYTE leftOrder, BYTE leftOrderToContract, const UINT* __restrict__ leftDim,
    BYTE rightOrder, BYTE rightOrderToContract, const UINT* __restrict__ rightDim);

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
