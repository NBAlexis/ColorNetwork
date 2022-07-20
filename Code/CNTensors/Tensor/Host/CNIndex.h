//=============================================================================
// FILENAME : CNIndex.h
// 
// DESCRIPTION:
// Typically, create index as:
// CNIndex(3, {CNOneIndex("a0", 4), CNOneIndex("a1", 4), CNOneIndex("a2", 4)});
// length is almost dim, except that, length can be smaller than dim
//
// REVISION[d-m-y]:
//  [02/05/2022 nbalexis]
//=============================================================================
#ifndef _CNINDEX_H_
#define _CNINDEX_H_

__BEGIN_NAMESPACE

class CNAPI CNIndex 
{
public:
	CNIndex()
		: m_iOrd(0)
		, m_uiVolume(0)
	{

	}

	CNIndex(const TArray<CNOneIndex>& indexes)
		: CNIndex(static_cast<UINT>(indexes.Num()), indexes.GetData())
	{
		
	}

	CNIndex(const class CNIndexBlockDetail& block);
	CNIndex(const class CNIndexBlockDetail& block1, const class CNIndexBlockDetail& block2);
	CNIndex(const TArray<class CNIndexBlockDetail>& blocks);

	CNIndex(const TArray<CCString>& names, const TArray<UINT>& lengths);

	CNIndex(UINT uiCount, const CNOneIndex* indexes);

	const CNIndex& operator=(const CNIndex& other);

	~CNIndex() {}

	const UINT* GetLengthes() const { return m_lstDim.GetData(); }
	const UINT* GetStrides() const { return m_lstStride.GetData(); }

	UBOOL GetBlock(const CNIndexBlock& ranges, TArray<UINT>& strides, TArray<UINT>& length, UINT& indexstart, UINT& volume) const;
	UBOOL GetBlock(const CNIndexBlock& ranges, CNIndexBlockDetail& outblock) const;
	UBOOL GetOneIndex(const CNOneIndexRange& range, UINT& addstartidx, UINT& stride, UINT& length) const;

	UBOOL Combine(const CNIndexName& combinedTo);
	UBOOL Split(const CNIndexName& tobesplit, UINT uiLength, const CNIndexName& newname);

	UBOOL Transpose(const CNIndexName* neworder, TArray<CNOneIndex>& toCreateNew, TArray<UINT>& dstStrides) const;

	void PrintIndex() const;

	UINT GetOrder() const { return m_iOrd; }
	UINT GetVolume() const { return m_uiVolume; }
	UINT GetDimOfOneOrder(UINT uiOrder) const 
	{
		if (uiOrder < static_cast<UINT>(m_lstDim.Num()))
		{
			return m_lstDim[uiOrder];
		}
		return 0;
	}

private:

	void RefreshTable();

	TArray<CNIndexName> m_lstIdx;
	TArray<UINT> m_lstDim;
	TArray<UINT> m_lstStride;
	UINT m_iOrd;
	UINT m_uiVolume;

	THashMap<QWORD, INT> m_indexTable;
};

__END_NAMESPACE

#endif//#ifndef _CNHOSTTENSOR_H_

//=============================================================================
// END OF FILE
//=============================================================================