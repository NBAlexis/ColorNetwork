//=============================================================================
// FILENAME : CNOneIndex.h
// 
// DESCRIPTION:
// Host tensor should be a template class without template function
//
// REVISION[d-m-y]:
//  [02/05/2022 nbalexis]
//=============================================================================
#ifndef _CNONEINDEX_H_
#define _CNONEINDEX_H_

__BEGIN_NAMESPACE

class CNAPI CNIndexName
{
public:

	CNIndexName()
		: m_ulId(0)
	{

	}

	CNIndexName(const CNIndexName& other)
		: m_ulId(other.m_ulId)
	{

	}

	CNIndexName(const ANSICHAR* name)
	{
		Rename(name);
	}

	void Rename(const ANSICHAR* name)
	{
		UINT len = static_cast<UINT>(appStrlen(name));
		memcpy(m_byName, name, len > 8 ? sizeof(ANSICHAR) * 8 : sizeof(ANSICHAR) * len);
		m_byName[7] = 0;
	}

	const ANSICHAR* c_str() const 
	{ 
		return (const ANSICHAR*)m_byName;
	}

	QWORD GetId() const
	{
		return m_ulId;
	}

	operator const ANSICHAR* () const { return (const ANSICHAR*)m_byName; }
	operator QWORD () const { return m_ulId; }
	const CNIndexName& operator=(const ANSICHAR* lpsz)
	{
		Rename(lpsz);
		return *this;
	}

	UBOOL operator==(const CNIndexName& s2) const
	{
		return GetId() == s2.GetId();
	}

private:

	union 
	{
		QWORD m_ulId;
		BYTE m_byName[8];
	};
};

class CNAPI CNOneIndex 
{
public:
	CNOneIndex()
		: m_uiLength(0)
		, m_sName(0)
	{

	}

	CNOneIndex(const ANSICHAR* name, UINT uiLength)
		: m_uiLength(uiLength)
		, m_sName(name)
	{

	}

	CNOneIndex(const CNOneIndex& other)
		: m_uiLength(other.m_uiLength)
		, m_sName(other.m_sName)
	{

	}

	const CNOneIndex& operator=(const CNOneIndex& other)
	{
		m_uiLength = other.m_uiLength;
		m_sName = other.m_sName;
		return *this;
	}

	UBOOL operator==(const CNOneIndex& other) const
	{
		return (m_sName == other.m_sName) && (m_uiLength == other.m_uiLength);
	}

	//========= general ==========
	UINT m_uiLength;

	CNIndexName m_sName;

};

/**
* 
*/
class CNAPI CNOneIndexRange
{
public:
	CNOneIndexRange()
		: m_iFrom(0)
		, m_iTo(0)
		, m_sName(0)
	{

	}

	CNOneIndexRange(const ANSICHAR* name)
		: m_iFrom(0)
		, m_iTo(-1)
		, m_sName(name)
	{

	}

	CNOneIndexRange(const ANSICHAR* name, INT idx)
		: m_iFrom(idx)
		, m_iTo(idx + 1)
		, m_sName(name)
	{

	}

	CNOneIndexRange(const ANSICHAR* name, INT iFrom, INT iTo)
		: m_iFrom(iFrom)
		, m_iTo(iTo)
		, m_sName(name)
	{

	}

	CNOneIndexRange(const CNOneIndexRange& other)
		: m_iFrom(other.m_iFrom)
		, m_iTo(other.m_iTo)
		, m_sName(other.m_sName)
	{

	}

	const CNOneIndexRange& operator=(const CNOneIndexRange& other)
	{
		m_iFrom = other.m_iFrom;
		m_iTo = other.m_iTo;
		m_sName = other.m_sName;
		return *this;
	}

	UBOOL operator==(const CNOneIndexRange& other) const
	{
		return (m_sName == other.m_sName) && (m_iFrom == other.m_iFrom) && (m_iTo == other.m_iTo);
	}

	//========= general ==========
	INT m_iFrom;
	INT m_iTo;

	CNIndexName m_sName;

};

/**
* A list of CNOneIndexRange
* 
*/
class CNAPI CNIndexBlock
{
public:
	CNIndexBlock(const TArray<CNOneIndexRange>& ranges)
		: m_lstRange(ranges)
	{

	}

	CNIndexBlock(const TArray<CNIndexName>& names)
	{
		for (INT i = 0; i < names.Num(); ++i)
		{
			m_lstRange.AddItem(CNOneIndexRange(names[i]));
		}
	}

	CNIndexBlock(UINT uiCount, const CNOneIndexRange* ranges)
	{
		for (UINT i = 0; i < uiCount; ++i)
		{
			m_lstRange.AddItem(ranges[i]);
		}
	}

	CNIndexBlock(UINT uiCount, const ANSICHAR* const* names)
	{
		for (UINT i = 0; i < uiCount; ++i)
		{
			m_lstRange.AddItem(CNOneIndexRange(names[i]));
		}
	}

	TArray<CNOneIndexRange> m_lstRange;

	CNIndexBlock GetLeft(INT iCount) const 
	{
		TArray<CNOneIndexRange> left;
		for (INT i = 0; i < iCount && i < m_lstRange.Num(); ++i)
		{
			left.AddItem(m_lstRange[i]);
		}
		return CNIndexBlock(left);
	}

	CNIndexBlock GetRight(INT iCount) const
	{
		TArray<CNOneIndexRange> right;
		for (INT i = (m_lstRange.Num() - iCount); i < m_lstRange.Num(); ++i)
		{
			right.AddItem(m_lstRange[i]);
		}
		return CNIndexBlock(right);
	}
};


__END_NAMESPACE

#endif//#ifndef _CNHOSTTENSOR_H_

//=============================================================================
// END OF FILE
//=============================================================================