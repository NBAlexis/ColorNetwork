//=============================================================================
// FILENAME : CNIndex.cpp
// 
// DESCRIPTION:
// 
//
// REVISION[d-m-y]:
//  [03/05/2022 nbale]
//=============================================================================
#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

CNIndex::CNIndex(UINT uiCount, const CNOneIndex* indexes)
	: m_iOrd(uiCount)
	, m_uiVolume(0)
{
	for (UINT i = 0; i < m_iOrd; ++i)
	{
		m_lstIdx.AddItem(indexes[i].m_sName);
		m_lstDim.AddItem(indexes[i].m_uiLength);
		m_lstStride.AddItem(1);
		m_indexTable.SetAt(indexes[i].m_sName, static_cast<INT>(i));
	}

	//m_lstStride[m_iDim - 1] = 1; already set to 1
	m_uiVolume = 1;
	for (UINT i = 1; i < m_iOrd; ++i)
	{
		m_uiVolume = m_uiVolume * m_lstDim[m_iOrd - i];
		m_lstStride[m_iOrd - i - 1] = m_uiVolume;
	}
	m_uiVolume = m_uiVolume * m_lstDim[0];
}

CNIndex::CNIndex(const TArray<CCString>& names, const TArray<UINT>& lengths)
{
	INT len = names.Num() > lengths.Num() ? lengths.Num() : names.Num();
	m_iOrd = static_cast<UINT>(len);
	for (INT i = 0; i < len; ++i)
	{
		m_lstIdx.AddItem(CNIndexName(names[i].c_str()));
		m_lstDim.AddItem(lengths[i]);
		m_lstStride.AddItem(1);
		m_indexTable.SetAt(m_lstIdx[i], i);
	}
	m_uiVolume = 1;
	for (UINT i = 1; i < m_iOrd; ++i)
	{
		m_uiVolume = m_uiVolume * m_lstDim[m_iOrd - i];
		m_lstStride[m_iOrd - i - 1] = m_uiVolume;
	}
	m_uiVolume = m_uiVolume * m_lstDim[0];
}

CNIndex::CNIndex(const CNIndexBlock& block)
{
	//TArray<CCString> names;
	//TArray<UINT> dims;
	//for (INT i = 0; i < block.m_lstRange.Num(); ++i)
	//{
	//	names.AddItem(block.m_lstRange[i].m_sName.c_str());
	//	dims.AddItem(block.m_lstRange[i].);
	//}
}

CNIndex::CNIndex(const CNIndexBlock& block1, const CNIndexBlock& block2)
{

}

CNIndex::CNIndex(const TArray<CNIndexBlock>& blocks)
{

}

const CNIndex& CNIndex::operator=(const CNIndex& other)
{
	m_iOrd = other.m_iOrd;
	m_lstIdx = other.m_lstIdx;
	m_lstDim = other.m_lstDim;
	m_lstStride = other.m_lstStride;
	m_indexTable = other.m_indexTable;
	m_uiVolume = other.m_uiVolume;
	return *this;
}

UBOOL CNIndex::Combine(const CNIndexName& combinedTo)
{
	INT order = -1;
	if (!m_indexTable.GetAt(combinedTo, order))
	{
		appWarning(_T("CNIndex::Combine: index name %s not found\n"), combinedTo.c_str());
		return FALSE;
	}

	if (order == static_cast<INT>(m_iOrd) - 1)
	{
		appWarning(_T("CNIndex::Combine: last index has nothing to combine\n"));
		return FALSE;
	}

	UINT newLength = m_lstDim[order] * m_lstDim[order + 1];
	UINT newStride = m_lstStride[order + 1];

	m_lstIdx.RemoveAt(order + 1);
	m_lstDim.RemoveAt(order + 1);
	m_lstStride.RemoveAt(order + 1);

	m_lstDim[order] = newLength;
	m_lstStride[order] = newStride;
	m_iOrd = m_iOrd - 1;
	RefreshTable();

	return TRUE;
}

UBOOL CNIndex::Split(const CNIndexName& tobesplit, UINT uiLength, const CNIndexName& newname)
{
	INT order = -1;
	if (!m_indexTable.GetAt(tobesplit, order))
	{
		appWarning(_T("CNIndex::Split: index name %s not found\n"), tobesplit.c_str());
		return FALSE;
	}
	if (m_indexTable.Exist(newname))
	{
		appWarning(_T("CNIndex::Split: duplicated new name %s\n"), newname.c_str());
		return FALSE;
	}
	UINT length1 = m_lstDim[order];
	UINT newLength2 = length1 / uiLength;
	if (uiLength * newLength2 != length1)
	{
		appWarning(_T("CNIndex::Split: try to split %d to %d\n"), length1, uiLength);
		return FALSE;
	}

	m_lstIdx.InsertAt(order + 1, newname);
	m_lstDim.InsertAt(order + 1, newLength2);
	m_lstStride.InsertAt(order + 1, m_lstStride[order]);

	m_lstDim[order] = uiLength;
	m_lstStride[order] = m_lstStride[order] * newLength2;

	RefreshTable();

	return TRUE;
}

UBOOL CNIndex::GetBlock(const CNIndexBlock& ranges, TArray<UINT>& strides, TArray<UINT>& length, UINT& indexstart) const
{
	indexstart = 0;
	strides.RemoveAll();
	length.RemoveAll();
	INT order = -1;
	for (INT i = 0; i < ranges.m_lstRange.Num(); ++i)
	{
		if (!m_indexTable.GetAt(ranges.m_lstRange[i].m_sName, order))
		{
			appWarning(_T("CNIndex::GetBlock: index name %s not found\n"), ranges.m_lstRange[i].m_sName.c_str());
			return FALSE;
		}

		INT iTo = ranges.m_lstRange[i].m_iTo;
		INT iFrom = ranges.m_lstRange[i].m_iFrom;
		if (iFrom < 0)
		{
			iFrom = 0;
		}
		if (iFrom >= static_cast<INT>(m_lstDim[order]))
		{
			iFrom = static_cast<INT>(m_lstDim[order]) - 1;
		}
		if (-1 == iTo)
		{
			iTo = m_lstDim[order];
		}

		if (iTo - ranges.m_lstRange[i].m_iFrom < 2)
		{
			indexstart = indexstart + static_cast<UINT>(iFrom) * m_lstStride[order];
		}
		else
		{
			const UINT uiTo = static_cast<UINT>(iTo);
			const UINT uiMaxIdx = uiTo > m_lstDim[order] ? m_lstDim[order] : uiTo;
			indexstart = indexstart + static_cast<UINT>(iFrom) * m_lstStride[order];
			strides.AddItem(m_lstStride[order]);
			length.AddItem(uiMaxIdx - static_cast<UINT>(iFrom));
		}
	}
	return TRUE;
}

UBOOL CNIndex::GetOneIndex(const CNOneIndexRange& range, UINT& addstartidx, UINT& stride, UINT& length) const
{
	INT order = -1;
	if (!m_indexTable.GetAt(range.m_sName, order))
	{
		appWarning(_T("CNIndex::GetBlock: index name %s not found\n"), range.m_sName.c_str());
		return FALSE;
	}

	INT iTo = range.m_iTo;
	INT iFrom = range.m_iFrom;
	if (iFrom < 0)
	{
		iFrom = 0;
	}
	if (iFrom >= static_cast<INT>(m_lstDim[order]))
	{
		iFrom = static_cast<INT>(m_lstDim[order]) - 1;
	}
	if (-1 == iTo)
	{
		iTo = m_lstDim[order];
	}

	if (iTo - range.m_iFrom < 2)
	{
		addstartidx = addstartidx + static_cast<UINT>(iFrom) * m_lstStride[order];
		stride = m_lstStride[order];
		length = 1;
	}
	else
	{
		const UINT uiTo = static_cast<UINT>(iTo);
		const UINT uiMaxIdx = uiTo > m_lstDim[order] ? m_lstDim[order] : uiTo;
		addstartidx = addstartidx + static_cast<UINT>(iFrom) * m_lstStride[order];
		stride = m_lstStride[order];
		length = uiMaxIdx - static_cast<UINT>(iFrom);
	}
	return TRUE;
}

void CNIndex::RefreshTable()
{
	m_indexTable.RemoveAll();
	for (UINT i = 0; i < m_iOrd; ++i)
	{
		m_indexTable.SetAt(m_lstIdx[i], static_cast<INT>(i));
	}
}

UBOOL CNIndex::Transpose(const CNIndexName* neworder, TArray<CNOneIndex>& toCreateNew, TArray<UINT>& dstStrides) const
{
	toCreateNew.RemoveAll();
	dstStrides.RemoveAll();

	for (UINT i = 0; i < m_iOrd; ++i)
	{
		INT order = -1;
		if (m_indexTable.GetAt(neworder[i], order))
		{
			toCreateNew.AddItem(CNOneIndex(neworder[i], m_lstDim[i]));
			dstStrides.AddItem(m_lstStride[i]);
		}
		else
		{
			appWarning(_T("CNIndex::Transpose: wrong index name:%s\n"), neworder[i].c_str());
			return FALSE;
		}
	}
	return TRUE;
}

void CNIndex::PrintIndex() const
{
	for (UINT i = 0; i < m_iOrd; ++i)
	{
		appGeneral(_T("(%s, %d, stride:%d) "), m_lstIdx[i].c_str(), m_lstDim[i], m_lstStride[i]);
	}
	appGeneral(_T("\n"));
}

__END_NAMESPACE

//=============================================================================
// END OF FILE
//=============================================================================
