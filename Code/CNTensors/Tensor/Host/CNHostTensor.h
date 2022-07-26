//=============================================================================
// FILENAME : CNHostTensor.h
// 
// DESCRIPTION:
// Host tensor should be a template class without template function
//
// REVISION:
//  [10/31/2021 nbalexis]
//=============================================================================
#ifndef _CNHOSTTENSOR_H_
#define _CNHOSTTENSOR_H_

#define __IMPLEMENT_HOST_TENSOR(type) \
template class CNHostTensor<type>;

#define __HOST_ONE_ELEMENT_INTERFACE_BLOCK(name) \
template<class Calc> \
UBOOL name(TCNDeviceTensorCommon<Calc>& calc, const CNIndexBlock& block) \
{ \
    UINT volume = 1; \
    UINT indexStart = 0; \
    TArray<UINT> strides; \
    TArray<UINT> lengths; \
    if (!m_Idx.GetBlock(block, strides, lengths, indexStart, volume)) \
    { \
        return FALSE; \
    } \
    m_cDeviceTensor.name(&calc, indexStart, strides.GetData(), lengths.GetData(), static_cast<BYTE>(lengths.Num())); \
    return TRUE; \
}\
template<class Calc> \
void name(TCNDeviceTensorCommon<Calc>& calc) \
{ \
    m_cDeviceTensor.name(&calc); \
}

#define __HOST_TWO_ELEMENT_INTERFACE_VALUE_BLOCK(name) \
template<class Calc, class Tsrc> \
UBOOL name(TCNDeviceTensorCommon<Calc>& calc, const CNIndexBlock& block, const Tsrc& v) \
{ \
    UINT indexStart = 0; \
    UINT volume = 1; \
    TArray<UINT> strides; \
    TArray<UINT> lengths; \
    if (!m_Idx.GetBlock(block, strides, lengths, indexStart, volume)) \
    { \
        return FALSE; \
    } \
    m_cDeviceTensor.name(&calc, v, indexStart, strides.GetData(), lengths.GetData(), static_cast<BYTE>(lengths.Num())); \
    return TRUE; \
} \
template<class Calc, class Tsrc> \
void name(TCNDeviceTensorCommon<Calc>& calc, const Tsrc& v) \
{ \
    m_cDeviceTensor.name(&calc, v); \
} 

#define __HOST_TWO_ELEMENT_INTERFACE_TENSOR_BLOCK(name) \
template<class Calc, class Tsrc> \
UBOOL name(TCNDeviceTensorCommon<Calc>& calc, const CNIndexBlock& block, const CNHostTensor<Tsrc>& v, const CNIndexBlock& srcblock) \
{ \
    UINT indexStart1 = 0; \
    TArray<UINT> strides1; \
    TArray<UINT> lengths1; \
    UINT indexStart2 = 0; \
    TArray<UINT> strides2; \
    TArray<UINT> lengths2; \
    if (!FixTwoTensorOperators(block, m_Idx, indexStart1, strides1, lengths1, srcblock, v.m_Idx, indexStart2, strides2, lengths2)) \
    { \
        return FALSE; \
    } \
    m_cDeviceTensor.name(&calc, v.m_cDeviceTensor.m_pDeviceDataBuffer, \
        indexStart1, strides1.GetData(), \
        indexStart2, strides2.GetData(), \
        lengths1.GetData(), static_cast<BYTE>(lengths1.Num())); \
    return TRUE; \
}

#define _HOST_MAKE_FRIENDS(typen) \
friend class CNHostTensor<typen>; \


__BEGIN_NAMESPACE

template<class T>
class __DLL_EXPORT CNHostTensor 
{
protected:

    struct PooledTensor
    {
        UBOOL m_bInUse;
        CNHostTensor<T>* m_pPointer;
    };

public:

    CNHostTensor()
        : m_pMother(NULL)
    {

    }

    ~CNHostTensor()
    {
        m_cDeviceTensor.Release();
        for (INT i = 0; i < m_Pool.Num(); ++i)
        {
            if (m_Pool[i].m_bInUse)
            {
                appWarning(_T("Pooled tensor in use when changing mother index!\n"));
            }
            appSafeDelete(m_Pool[i].m_pPointer);
        }
        m_Pool.RemoveAll();
    }

#pragma region Create and reshape

public:

    void Create(const CNIndex& index)
    {
        m_Idx = index;
        m_cDeviceTensor.CreateEmpty(m_Idx.GetVolume());
        for (INT i = 0; i < m_Pool.Num(); ++i)
        {
            if (m_Pool[i].m_bInUse)
            {
                appWarning(_T("Pooled tensor in use when changing mother index!\n"));
            }
            m_Pool[i].m_pPointer->Create(m_Idx);
        }
    }

    void Release()
    {
        m_cDeviceTensor.Release();
        for (INT i = 0; i < m_Pool.Num(); ++i)
        {
            if (m_Pool[i].m_bInUse)
            {
                appWarning(_T("Pooled tensor in use when changing mother index!\n"));
            }
            appSafeDelete(m_Pool[i].m_pPointer);
        }
        m_Pool.RemoveAll();
    }

    template<class Calc>
    UBOOL Transpose(TCNDeviceTensorCommon<Calc>& calc, CNHostTensor<T>& taget, const CNIndexName* neworder) const
    {
        TArray<CNOneIndex> newIdx;
        TArray<UINT> strides;
        if (!m_Idx.Transpose(neworder, newIdx, strides))
        {
            return FALSE;
        }

        CNIndex targetIdx(m_Idx.GetOrder(), newIdx.GetData());
        taget.Create(targetIdx);
        taget.m_cDeviceTensor.Set(
            &calc, 
            m_cDeviceTensor.m_pDeviceDataBuffer,
            0, 
            targetIdx.GetStrides(),
            0, 
            strides.GetData(),
            targetIdx.GetLengthes(),
            m_Idx.GetOrder());

        return TRUE;
    }

    /**
    * t1[a,b',d] = t2[a,(b,c),d]
    * if you need 
    * t1[a,b',c] = t2[a,(b,d),c]
    * transpose first
    */
    UBOOL Combine(const CNIndexName& combinedTo)
    {
        return m_Idx.Combine(combinedTo);
    }

    /**
    * t1[a,b1,b2,c] = t2[a,(b1,b2),d]
    * uiLength is length of b1
    */
    UBOOL Split(const CNIndexName& tobesplit, UINT uiLength, const CNIndexName& newname)
    {
        return m_Idx.Split(tobesplit, uiLength, newname);
    }

    UINT GetVolume() const { return m_Idx.GetVolume(); }
    UINT GetOrder() const { return m_Idx.GetOrder(); }
    UINT GetDimOfOneOrder(UINT uiOrder) const { return m_Idx.GetDimOfOneOrder(uiOrder); }

#pragma endregion

#pragma region Blocked One Tensor Operation

public:

    template<class Calc, class Tsrc>
    void Set(TCNDeviceTensorCommon<Calc>& calc, const Tsrc& v)
    {
        m_cDeviceTensor.Set(&calc, v);
    }

    template<class Calc, class Tsrc>
    UBOOL Set(TCNDeviceTensorCommon<Calc>& calc, const CNIndexBlock& block, const Tsrc& v)
    {
        UINT indexStart = 0;
        UINT volume = 1;
        TArray<UINT> strides;
        TArray<UINT> lengths;
        if (!m_Idx.GetBlock(block, strides, lengths, indexStart, volume))
        {
            return FALSE;
        }
        m_cDeviceTensor.Set(&calc, v, indexStart, strides.GetData(), lengths.GetData(), static_cast<BYTE>(lengths.Num()));
        return TRUE;
    }

    template<class Calc, class Tsrc>
    UBOOL SetColumn(TCNDeviceTensorCommon<Calc>& calc, BYTE indexCount, UINT uiIndexStart, const UINT* strides, const UINT* lengths, const Tsrc& v)
    {
        m_cDeviceTensor.Set(&calc, v, uiIndexStart, strides, lengths, indexCount);
        return TRUE;
    }

    template<class Calc, class Tsrc>
    UBOOL SetColumn(TCNDeviceTensorCommon<Calc>& calc, 
        BYTE indexCount, 
        UINT uiIndexStart1, 
        const UINT* strides1, 
        UINT uiIndexStart2,
        const UINT* strides2,
        const UINT* lengths,
        const UINT hostBufferVolume,
        const Tsrc* hostBuffer)
    {
        Tsrc* devicceBuffer = (Tsrc*)appGetSmallDeviceBuffer(hostBufferVolume * sizeof(Tsrc));
        _memcpy_hd(devicceBuffer, hostBuffer, hostBufferVolume * sizeof(Tsrc));
        m_cDeviceTensor.Set(&calc, devicceBuffer,
            uiIndexStart1, strides1,
            uiIndexStart2, strides2,
            lengths, indexCount);
        return TRUE;
    }

    template<class Calc>
    void Random(TCNDeviceTensorCommon<Calc>& calc, UINT uiRandomType)
    {
        m_cDeviceTensor.Random(&calc, uiRandomType);
    }

    template<class Calc>
    UBOOL Random(TCNDeviceTensorCommon<Calc>& calc, UINT uiRandomType, const CNIndexBlock& block)
    {
        UINT indexStart = 0;
        UINT volume = 1;
        TArray<UINT> strides;
        TArray<UINT> lengths;
        if (!m_Idx.GetBlock(block, strides, lengths, indexStart, volume))
        {
            return FALSE;
        }
        m_cDeviceTensor.Random(&calc, uiRandomType, indexStart, strides.GetData(), lengths.GetData(), static_cast<BYTE>(lengths.Num()));
        return TRUE;
    }

    __OVER_ALL_ONE_OP(__HOST_ONE_ELEMENT_INTERFACE_BLOCK)

    __OVER_ALL_TWO_OP(__HOST_TWO_ELEMENT_INTERFACE_VALUE_BLOCK)

#pragma endregion

#pragma region Blocked One Tensor Operation

protected:

    /**
    * use the blocks to generate strides and lengths for different tensor.
    * For example left[a1,a2]=right[b1,b2]
    * it searches the index of left, and find the range of a1,a2
    * at the same time, searches the index of right, and find the range of b1, b2
    * aligen each other to generate strides for each tensor
    * 
    * Note that, it makes sure length1[i] = length2[i]
    */
    static UBOOL FixTwoTensorOperators(
        const CNIndexBlock& block, const CNIndex& idx, UINT& idxstart1, TArray<UINT>& stride1, TArray<UINT>& length1,
        const CNIndexBlock& srcblock, const CNIndex& srcidx, UINT& idxstart2, TArray<UINT>& stride2, TArray<UINT>& length2)
    {
        UINT volume = 1;
        if (!idx.GetBlock(block, stride1, length1, idxstart1, volume))
        {
            return FALSE;
        }
        if (!srcidx.GetBlock(srcblock, stride2, length2, idxstart2, volume))
        {
            return FALSE;
        }
        if (length1.Num() != length2.Num())
        {
            appWarning(_T("CNHostTensor::FixTwoTensorOperators, order is not the same!\n"));
            return FALSE;
        }
        for (INT i = 0; i < length1.Num(); ++i)
        {
            if (length1[i] != length2[i])
            {
                appDetailed(_T("CNHostTensor::FixTwoTensorOperators dim not same idx is dest[%s], src[%s], length = %d, %d, set to be the smaller\n"), block.m_lstRange[i].m_sName.c_str(), srcblock.m_lstRange[i].m_sName.c_str(), length1[i], length2[i]);
                if (length1[i] > length2[i])
                {
                    length1[i] = length2[i];
                }
                else
                {
                    length2[i] = length1[i];
                }
            }
        }
        return TRUE;
    }

public:

    template<class Calc, class Tsrc>
    UBOOL Set(TCNDeviceTensorCommon<Calc>& calc, const CNIndexBlock& block, const CNHostTensor<Tsrc>& v, const CNIndexBlock& srcblock)
    {
        UINT indexStart1 = 0;
        TArray<UINT> strides1;
        TArray<UINT> lengths1;
        UINT indexStart2 = 0;
        TArray<UINT> strides2;
        TArray<UINT> lengths2;
        if (!FixTwoTensorOperators(block, m_Idx, indexStart1, strides1, lengths1, srcblock, v.m_Idx, indexStart2, strides2, lengths2))
        {
            return FALSE;
        }
        m_cDeviceTensor.Set(&calc, v.m_cDeviceTensor.m_pDeviceDataBuffer,
            indexStart1, strides1.GetData(),
            indexStart2, strides2.GetData(),
            lengths1.GetData(), static_cast<BYTE>(lengths1.Num()));
    }

    template <class Calc, class Tsrc>
    UBOOL Axpy(
        TCNDeviceTensorCommon<Calc>& calc,
        const CNIndexBlock& block,
        const T& v,
        const CNHostTensor<Tsrc>& src,
        const CNIndexBlock& srcblock)
    {

        UINT indexStart1 = 0;
        TArray<UINT> strides1;
        TArray<UINT> lengths1;
        UINT indexStart2 = 0;
        TArray<UINT> strides2;
        TArray<UINT> lengths2;
        if (!FixTwoTensorOperators(block, m_Idx, indexStart1, strides1, lengths1, srcblock, src.m_Idx, indexStart2, strides2, lengths2))
        {
            return FALSE;
        }
        m_cDeviceTensor.Axpy(&calc, v, src.m_cDeviceTensor.m_pDeviceDataBuffer,
            indexStart1, strides1.GetData(),
            indexStart2, strides2.GetData(),
            lengths1.GetData(), static_cast<BYTE>(lengths1.Num()));
        return TRUE;
    }

    __OVER_ALL_TWO_OP(__HOST_TWO_ELEMENT_INTERFACE_TENSOR_BLOCK)

#pragma endregion

#pragma region IO

public:

    void PrintIndex() const
    {
        m_Idx.PrintIndex();
    }

    void PrintContent(UINT uiEachLine, UINT uiLines = 0) const
    {
        m_cDeviceTensor.DebugPrint(uiEachLine, uiLines);
    }

#pragma endregion

#pragma region reduce

    /**
    * TODO: Create a temp tensor to do the reduce sum
    */
    template <class Calc>
    T ReduceSum(TCNDeviceTensorContraction<Calc>& calc)
    {
        return m_cDeviceTensor.ReduceSum(&calc);
    }

    template <class Calc>
    T ReduceProd(TCNDeviceTensorContraction<Calc>& calc)
    {
        return m_cDeviceTensor.ReduceProd(&calc);
    }

    /**
    * target[dstblock] = sum_a me[block, a]
    */
    template<class Calc>
    UBOOL Sum(TCNDeviceTensorContraction<Calc>& calc, CNHostTensor<T>& target, const CNIndexBlock& dstblock, const CNIndexBlock& block, const CNOneIndexRange& toSum) const
    {
        UINT indexStart1 = 0;
        TArray<UINT> strides1;
        TArray<UINT> lengths1;
        UINT indexStart2 = 0;
        TArray<UINT> strides2;
        TArray<UINT> lengths2;
        if (!FixTwoTensorOperators(block, m_Idx, indexStart1, strides1, lengths1, dstblock, target.m_Idx, indexStart2, strides2, lengths2))
        {
            return FALSE;
        }
        UINT sumstride = 1;
        UINT sumlength = 1;
        if (!m_Idx.GetOneIndex(toSum, indexStart1, sumstride, sumlength))
        {
            return FALSE;
        }

        target.m_cDeviceTensor.Sum(&calc, m_cDeviceTensor.m_pDeviceDataBuffer, 
            indexStart2, strides2.GetData(), 
            indexStart1, strides1.GetData(),
            lengths1.GetData(), static_cast<BYTE>(lengths1.Num()), 
            sumlength, sumstride);

        return TRUE;
    }

    template<class Calc>
    UBOOL Prod(TCNDeviceTensorContraction<Calc>& calc, CNHostTensor<T>& target, const CNIndexBlock& dstblock, const CNIndexBlock& block, const CNOneIndexRange& toSum) const
    {
        UINT indexStart1 = 0;
        TArray<UINT> strides1;
        TArray<UINT> lengths1;
        UINT indexStart2 = 0;
        TArray<UINT> strides2;
        TArray<UINT> lengths2;
        if (!FixTwoTensorOperators(block, m_Idx, indexStart1, strides1, lengths1, dstblock, target.m_Idx, indexStart2, strides2, lengths2))
        {
            return FALSE;
        }
        UINT sumstride = 1;
        UINT sumlength = 1;
        if (!m_Idx.GetOneIndex(toSum, indexStart1, sumstride, sumlength))
        {
            return FALSE;
        }

        target.m_cDeviceTensor.Prod(&calc, m_cDeviceTensor.m_pDeviceDataBuffer,
            indexStart2, strides2.GetData(),
            indexStart1, strides1.GetData(),
            lengths1.GetData(), static_cast<BYTE>(lengths1.Num()),
            sumlength, sumstride);

        return TRUE;
    }

#pragma endregion

#pragma region Contraction

    /**
    * me[block] = sum _i left[lblock, lcontractblock[i]] * right[rblock, rcontractblock[i]]
    * note that when the index is mixed, for example:
    * me[a1, b1, a2, b2] = sum _c left[a1, a2, c] right[b1, b2, c]
    * the result will arrange the block as (a1,a2,b1,b2)
    */
    template<class Calc, class rightT>
    UBOOL Contraction(TCNDeviceTensorContraction<Calc>& calc,
        const CNHostTensor<T>& left, const CNHostTensor<T>& right, const CNIndexBlock& block,
        const CNIndexBlock& lblock, const CNIndexBlock& rblock, 
        const CNOneIndexRange& lcontractblock, const CNOneIndexRange& rcontractblock,
        UBOOL bConjugate)
    {
        UINT volume = 1;
        UINT indexStart1 = 0;
        TArray<UINT> strides1;
        TArray<UINT> lengths1;
        UINT indexStart2 = 0;
        TArray<UINT> strides2;
        TArray<UINT> lengths2;
        if (!left.m_Idx.GetBlock(lblock, strides1, lengths1, indexStart1, volume))
        {
            return FALSE;
        }
        if (!right.m_Idx.GetBlock(rblock, strides2, lengths2, indexStart2, volume))
        {
            return FALSE;
        }

        UINT sumstride1 = 1;
        UINT sumlength1 = 1;
        if (!left.m_Idx.GetOneIndex(lcontractblock, indexStart1, sumstride1, sumlength1))
        {
            return FALSE;
        }
        UINT sumstride2 = 1;
        UINT sumlength2 = 1;
        if (!right.m_Idx.GetOneIndex(rcontractblock, indexStart2, sumstride2, sumlength2))
        {
            return FALSE;
        }
        UINT indexStartme = 0;
        TArray<UINT> stridesme;
        TArray<UINT> lengthsme;
        if (!m_Idx.GetBlock(block, stridesme, lengthsme, indexStartme, volume))
        {
            return FALSE;
        }
        if (lengthsme.Num() != lengths1.Num() + lengths2.Num())
        {
            appWarning(_T("CNHostTensor::Contraction, number of index not same %d != %d + %d\n"), 
                lengthsme.Num(), lengths1.Num(), lengths2.Num());
            return FALSE;
        }

        for (INT i = 0; i < lengthsme.Num(); ++i)
        {
            if (i < lengths1.Num())
            {
                if (lengthsme[i] > lengths1[i])
                {
                    lengthsme[i] = lengths1[i];
                }
            }
            else
            {
                if (lengthsme[i] > lengths2[i - lengths1.Num()])
                {
                    lengthsme[i] = lengths2[i - lengths1.Num()];
                }
            }
        }
        if (sumlength1 > sumlength2)
        {
            sumlength1 = sumlength2;
        }
        m_cDeviceTensor.Contraction(&calc, 
            left.m_cDeviceTensor.m_pDeviceDataBuffer, 
            right.m_cDeviceTensor.m_pDeviceDataBuffer,
            indexStartme, stridesme.GetData(),
            indexStart1, indexStart2,
            strides1.GetData(), strides2.GetData(), lengthsme.GetData(),
            static_cast<BYTE>(lengthsme.Num()), 
            static_cast<BYTE>(lengthsme.Num() - lengths1.Num()),
            sumlength1, sumstride1, sumstride2,
            bConjugate);

        return TRUE;
    }

    /**
    * same as above, but contract can be multi-index
    * the range of sum is aligned acoording to left block
    * 
    */
    template<class Calc, class rightT>
    UBOOL Contraction(TCNDeviceTensorContraction<Calc>& calc,
        const CNHostTensor<T>& left, const CNHostTensor<T>& right, const CNIndexBlock& block,
        const CNIndexBlock& lblock, const CNIndexBlock& rblock,
        const CNIndexBlock& lcontractblock, const CNIndexBlock& rcontractblock,
        UBOOL bConjugate)
    {
        UINT volume = 1;
        UINT indexStart1 = 0;
        UINT indexStart2 = 0;
        UINT sumIndexStart1 = 0;
        UINT sumIndexStart2 = 0;
        TArray<UINT> strides1;
        TArray<UINT> lengths1;
        TArray<UINT> strides2;
        TArray<UINT> lengths2;
        TArray<UINT> sumstrideleft;
        TArray<UINT> sumstrideright;
        TArray<UINT> sumlengths1;
        TArray<UINT> sumlengths2;

        if (!left.m_Idx.GetBlock(lblock, strides1, lengths1, indexStart1, volume))
        {
            return FALSE;
        }
        if (!right.m_Idx.GetBlock(rblock, strides2, lengths2, indexStart2, volume))
        {
            return FALSE;
        }

        if (!FixTwoTensorOperators(
            lcontractblock, left.m_Idx, sumIndexStart1, sumstrideleft, sumlengths1,
            rcontractblock, right.m_Idx, sumIndexStart2, sumstrideright, sumlengths2))
        {
            return FALSE;
        }

        if (sumlengths1.Num() > _CN_CONTRACTION_INDEX_COUNT_ONE_TIME)
        {
            return FALSE;
        }

        indexStart1 += sumIndexStart1;
        indexStart2 += sumIndexStart2;

        UINT indexStartme = 0;
        TArray<UINT> stridesme;
        TArray<UINT> lengthsme;
        if (!m_Idx.GetBlock(block, stridesme, lengthsme, indexStartme, volume))
        {
            return FALSE;
        }
        if (lengthsme.Num() != lengths1.Num() + lengths2.Num())
        {
            appWarning(_T("CNHostTensor::Contraction, number of index not same %d != %d + %d\n"),
                lengthsme.Num(), lengths1.Num(), lengths2.Num());
            return FALSE;
        }

        for (INT i = 0; i < lengthsme.Num(); ++i)
        {
            if (i < lengths1.Num())
            {
                if (lengthsme[i] > lengths1[i])
                {
                    lengthsme[i] = lengths1[i];
                }
            }
            else
            {
                if (lengthsme[i] > lengths2[i - lengths1.Num()])
                {
                    lengthsme[i] = lengths2[i - lengths1.Num()];
                }
            }
        }

        m_cDeviceTensor.Contraction(&calc,
            left.m_cDeviceTensor.m_pDeviceDataBuffer,
            right.m_cDeviceTensor.m_pDeviceDataBuffer,
            indexStartme, stridesme.GetData(),
            indexStart1, indexStart2,
            strides1.GetData(), strides2.GetData(), lengthsme.GetData(),
            static_cast<BYTE>(lengthsme.Num()),
            static_cast<BYTE>(lengthsme.Num() - lengths1.Num()),
            sumstrideleft.GetData(), sumstrideright.GetData(), sumlengths1.GetData(),
            static_cast<BYTE>(sumlengths1.Num()),
            bConjugate);

        return TRUE;
    }

#pragma endregion

    //I cann't see the reason to implement this...
#if 0
#pragma region Matrix style contraction

    /**
    * y[i] = A.x[j], where y[i] is a vector or the i-th line of a matrix, x[j] is a vector or the j-th line of a matrix
    * Which is t[i,m] = sum _n A[m,n] B[j,n]
    *
    * that is:
    * ty[i, m0+m1*l1+m2*l2] = sum
    */
    template<class Calc, class rightT>
    static UBOOL MatrixMxM(TCNDeviceTensorContraction<Calc>& calc,
        CNHostTensor<T>** newRes,
        const CNHostTensor<T>& A, const CNHostTensor<T>& B,
        const CNIndexBlock& blockI, const CNIndexBlock& blockM,
        const CNIndexBlock& blockN, const CNIndexBlock& blockJ)
    {
        UINT indexStart1 = 0;
        TArray<UINT> strides1;
        TArray<UINT> lengths1;
        UINT indexStart2 = 0;
        TArray<UINT> strides2;
        TArray<UINT> lengths2;
        if (!FixTwoTensorOperators(
            blockM, A.m_Idx, indexStart1, strides1, lengths1,
            blockJ, B.m_Idx, indexStart2, strides2, lengths2))
        {
            *newRes = NULL;
            return FALSE;
        }

        //CNIndex
        newRes = new CNHostTensor<T>();
        //resIndex = CNIndex();
    }

    /**
    * y[i] = A.x[j], where y[i] is a vector or the i-th line of a matrix, x[j] is a vector or the j-th line of a matrix
    * Which is ty[i,m] = sum _n A[m,n] x[j,n]
    * 
    * that is:
    * ty[i, m0+m1*l1+m2*l2] = sum 
    */
    template<class Calc, class rightT>
    UBOOL MatrixMxV(TCNDeviceTensorContraction<Calc>& calc,
        const CNHostTensor<T>& left, const CNHostTensor<T>& right,
        const CNIndexBlock& blockI, const CNIndexBlock& blockj,
        const CNIndexBlock& blockM, const CNIndexBlock& blockN)
    {

    }

    /**
    * dot: sum _i x*[i] y[j]
    */
    template<class Calc, class rightT>
    UBOOL MatrixVxV(TCNDeviceTensorContraction<Calc>& calc,
        const CNHostTensor<T>& left, const CNHostTensor<T>& right,
        const CNIndexBlock& blockI, const CNIndexBlock& blockj,
        const CNIndexBlock& blockM, const CNIndexBlock& blockN)
    {

    }

#pragma endregion
#endif

#pragma region Copy and Pool

protected:

    /**
    * To use a pool, create a tensor as mother
    */
    TArray<PooledTensor> m_Pool;
    CNHostTensor<T>* m_pMother;

    void ReleasePooledObject(CNHostTensor<T>* obj)
    {
        for (INT i = 0; i < m_Pool.Num(); ++i)
        {
            if (m_Pool[i].m_pPointer == obj)
            {
                m_Pool[i].m_bInUse = FALSE;
                return;
            }
        }
    }

public:

    void CopyTo(CNHostTensor<T>* target) const
    {
        target->Create(m_Idx);
        _memcpy_dd(target->m_cDeviceTensor.m_pDeviceDataBuffer, m_cDeviceTensor.m_pDeviceDataBuffer, sizeof(T) * m_cDeviceTensor.m_uiTotalSize);
    }

    CNHostTensor<T>* GetNewWithSameSize() const
    {
        CNHostTensor<T>* ret = new CNHostTensor<T>();
        ret->Create(m_Idx);
        return ret;
    }

    CNHostTensor<T>* GetCopy() const
    {
        CNHostTensor<T>* ret = GetNewWithSameSize();
        CopyTo(ret);
        return ret;
    }

    CNHostTensor<T>* GetPooledCopy()
    {
        CNHostTensor<T>* newone = GetPooled();
        CopyTo(newone);
        return newone;
    }

    CNHostTensor<T>* GetPooled()
    {
        for (INT i = 0; i < m_Pool.Num(); ++i)
        {
            if (!m_Pool[i].m_bInUse)
            {
                m_Pool[i].m_bInUse = TRUE;
                return m_Pool[i].m_pPointer;
            }
        }

        CNHostTensor<T>* newone = GetNewWithSameSize();
        newone->m_pMother = this;
        m_Pool.AddItem({TRUE, newone});
        return newone;
    }

    void ReleaseMe()
    {
        if (NULL != m_pMother)
        {
            m_pMother->ReleasePooledObject(this);
        }
    }

#pragma endregion

#pragma region IO

    UBOOL CreateWithFile(const CCString& sFileName)
    {
        UINT uiTotalSize = 0;
        BYTE* file = CFile::ReadAllBytes(sFileName.c_str(), uiTotalSize);
        if (0 == uiTotalSize)
        {
            appCrucial(_T("Tensor load failed: %s\n"), sFileName.c_str());
            return FALSE;
        }

        UINT uiOrder = 0;
        memcpy(&uiOrder, file, sizeof(UINT));
        TArray<UINT> dims;
        TArray<CNIndexName> names;
        UINT uiDim = 0;
        QWORD indexName = 0;
        for (UINT i = 0; i < uiOrder; ++i)
        {
            memcpy(&uiDim, file + sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * i, sizeof(UINT));
            memcpy(&indexName, file + sizeof(UINT) * 2 + (sizeof(UINT) + sizeof(QWORD)) * i, sizeof(QWORD));
            dims.AddItem(uiDim);
            names.AddItem(CNIndexName(indexName));
        }

        m_Idx = CNIndex(names, dims);
        Create(m_Idx);

        const BYTE byType = file[sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * m_Idx.GetOrder()];
        const UINT uiStride = GetTypeSizeOf(byType);
        const UINT uiStart = sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * m_Idx.GetOrder() + 4;

        if (byType != TensorRtti(m_cDeviceTensor.m_pDeviceDataBuffer))
        {
            static const ANSICHAR* namestring[7] =
            {
                "Unknown",
                "SBYTE",
                "INT",
                "FLOAT",
                "DOUBLE",
                "_SComplex",
                "_DComplex",
            };
            appWarning(_T("Reading a tensor from file %s(%s) with WRONG TYPE(%s), implicity cast is applied.\n"), 
                sFileName.c_str(), namestring[byType], namestring[TensorRtti(m_cDeviceTensor.m_pDeviceDataBuffer)]);
        }

        T* writeBuffer = (T*)malloc(sizeof(T) * m_Idx.GetVolume());
        if (NULL == writeBuffer)
        {
            appCrucial(_T("Tensor load failed: %s\n"), sFileName.c_str());
            return FALSE;
        }
        for (UINT i = 0; i < m_Idx.GetVolume(); ++i)
        {
            TensorTypeConvert(writeBuffer[i], byType, file + uiStart + uiStride * i);
            //LogValue(writeBuffer[i]);
            //appGeneral(_T("\n"));
        }
        m_cDeviceTensor.CreateEmpty(m_Idx.GetVolume());
        m_cDeviceTensor.CopyIn((BYTE*)writeBuffer);

        return TRUE;
    }

    UBOOL SaveToFile(const CCString& sFileName) const
    {
        const UINT totalSize = sizeof(UINT) 
            + (sizeof(UINT) + sizeof(QWORD)) * m_Idx.GetOrder() 
            + sizeof(BYTE) * 4
            + sizeof(T) * m_Idx.GetVolume();
        BYTE* data = (BYTE*)malloc(totalSize);
        if (NULL == data)
        {
            appCrucial(_T("Tensor save failed: %s\n"), sFileName.c_str());
            return FALSE;
        }

        const UINT uiOrder = m_Idx.GetOrder();
        memcpy(data, &uiOrder, sizeof(UINT));
        for (UINT i = 0; i < uiOrder; ++i)
        {
            UINT uiLength = m_Idx.GetLength(i);
            QWORD qwName = m_Idx.GetName(i);
            memcpy(data + sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * i, &uiLength, sizeof(UINT));
            memcpy(data + sizeof(UINT) * 2 + (sizeof(UINT) + sizeof(QWORD)) * i, &qwName, sizeof(QWORD));
        }

        BYTE byTensorType = TensorRtti(m_cDeviceTensor.m_pDeviceDataBuffer);
        memcpy(data + sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * uiOrder, &byTensorType, sizeof(BYTE));
        byTensorType = 0;
        memcpy(data + 1 + sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * uiOrder, &byTensorType, sizeof(BYTE));
        memcpy(data + 2 + sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * uiOrder, &byTensorType, sizeof(BYTE));
        memcpy(data + 3 + sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * uiOrder, &byTensorType, sizeof(BYTE));

        UINT uiBufferSize = sizeof(T) * m_Idx.GetVolume();
        m_cDeviceTensor.CopyOut(data + 4 + sizeof(UINT) + (sizeof(UINT) + sizeof(QWORD)) * uiOrder);

        if (!CFile::WriteAllBytes(sFileName.c_str(), data, totalSize))
        {
            appCrucial(_T("Tensor save failed: %s\n"), sFileName.c_str());
            appSafeFree(data);
            return FALSE;
        }

        appSafeFree(data);

        return TRUE;
    }

#pragma endregion

public:

    __OVER_ALL_TYPE_ONE(_HOST_MAKE_FRIENDS)

protected:

    CNDeviceTensor<T> m_cDeviceTensor;
    CNIndex m_Idx;
};


__END_NAMESPACE

#endif//#ifndef _CNHOSTTENSOR_H_

//=============================================================================
// END OF FILE
//=============================================================================