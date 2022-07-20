//=============================================================================
// FILENAME : CNMatrixSVD.h
// 
// DESCRIPTION:
// treat t1 as a matrix
// t1 = v.s.u, where v,u are orthognal matrices, s is diagnal matrix
// Note that, this is not tensor SVD, which decompse:
// t1[a,b,c,d] = s[A,B,C,D] u1[a,A] u2[b,B] u3[c,C] u4[d,D], where u are orthognal matrices, s is diagnal tensor
//
// REVISION[d-m-y]:
//  [15/07/2022 nbalexis]
//=============================================================================
#ifndef _CNMATRIXSVD_H_
#define _CNMATRIXSVD_H_

__BEGIN_NAMESPACE

template<class T>
class CNAPI CNMatrixSVD
{
public:
	CNMatrixSVD()
	{

	}

	//template <class T>
	//UBOOL Lanczos(const CNHostTensor<T>& t1, const CNIndexBlock& left, const CNIndexBlock& right, CNHostTensor<T>** q, UINT uiK)
	//{

	//}

	/**
	* use Lanzcos method to calculate
	* t1[left, right] = v[left,mid1] s[mid1,mid2] u[mid2, right], where order(mid1)=order(mid2)=truncate
	*/
	template<class CalcCommon, class CalcContraction>
	UBOOL DecomposeLanczos(
		TCNDeviceTensorCommon<CalcCommon>& calcCommon,
		TCNDeviceTensorContraction<CalcContraction>& calcContract,
		const CNHostTensor<T>& A, const CNIndexBlock& left, const CNIndexBlock& right, CNHostTensor<T>** v, CNHostTensor<T>** s, CNHostTensor<T>** u, UINT uiTruncate, CNIndexName leftName, CNIndexName rightName)
	{
		//=============================================
		//1 - first step, figure out the size of the tensor
		CNIndexBlockDetail leftblock; 
		CNIndexBlockDetail rightblock;
		if (!A.m_Idx.GetBlock(left, leftblock))
		{
			return FALSE;
		}

		if (!A.m_Idx.GetBlock(right, rightblock))
		{
			return FALSE;
		}

		CNIndexBlockDetail leftIdx(leftName, uiTruncate);
		CNIndexBlockDetail rightIdx(rightName, uiTruncate);

		CNIndex vindex(leftblock, leftIdx);
		CNIndex sindex(leftIdx, rightIdx);
		CNIndex uindex(rightIdx, rightblock);

		CNIndex pindex(CNIndexBlockDetail(leftName, 1), leftblock);
		CNIndex rindex(CNIndexBlockDetail(rightName, 1), rightblock);
		CNIndexBlock pblock = CNIndexBlock(CNIndexBlock(1, &leftName), left);
		CNIndexBlock rblock = CNIndexBlock(CNIndexBlock(1, &rightName), right);

		m_p.Create(pindex);
		m_psum.Create(pindex);
		m_r.Create(rindex);
		m_rsum.Create(rindex);

		CNHostTensor<T>* pv = new CNHostTensor<T>();
		CNHostTensor<T>* ps = new CNHostTensor<T>();
		CNHostTensor<T>* pu = new CNHostTensor<T>();
		*v = pv;
		*s = ps;
		*u = pu;
		pv->Create(vindex);
		ps->Create(sindex);
		pu->Create(uindex);

		CNIndexBlock leftVectorBlock = CNIndexBlock(leftName, 0, leftblock.m_lstNames);
		CNIndexBlock rightVectorBlock = CNIndexBlock(rightName, 0, rightblock.m_lstNames);
		CNIndexBlock leftVectorIndex = CNIndexBlock(1, &leftName);
		CNIndexBlock rightVectorIndex = CNIndexBlock(1, &rightName);

		//=============================================
		//2 - initial v1 as random unit vector
		pv->Random(calcCommon, 0, leftVectorBlock);
		m_psum.Set(calcCommon, left, *pv, leftVectorBlock);
		m_psum.ConjMul(calcCommon, left, m_psum, left);
		T sum = m_psum.ReduceSum(calcContract);
		const T one = _One(sum);
		pv->Mul(calcCommon, leftVectorBlock, _Div(one, sum));
		
		TArray<T> ai;
		TArray<T> bi;
		//=============================================
		//3 - iteration
		//for i = 1 to k do
		//  if i > 1 then
		//      vi = p / b(i - 1).
		//  end if
		//  r = A vi - b(i-1)u(i-1). b0=0 and u0=0
		//  ai = ||r||
		//  ui = r / ai.
		//  if i < k then
		//      p = Ad ui - aivi
		//      bi = || p ||
		//  end if
		//end for
		for (UINT i = 0; i < uiTruncate; ++i)
		{
			leftVectorBlock.m_lstRange[0].SetAsElementIndex(i);

			if (i > 1)
			{
				pv->Set(calcCommon, leftVectorBlock, m_p, left);
				pv->Mul(calcCommon, leftVectorBlock, _Div(one, bi[i - 1]));
			}
			
			//m_r = sum _left A[right, left] pv[index, left]
			leftVectorIndex.m_lstRange[0].SetAsElementIndex(i);
			m_r.Contraction(calcContract, *pv, A, rblock, leftVectorIndex, right, left, left, FALSE);

			//m_r = m_r - b(i-1)u(i-1)
			if (i > 1)
			{
				rightVectorBlock.m_lstRange[0].SetAsElementIndex(i - 1);
				m_r.Axpy(calcCommon, rblock, _Oppo(bi[i - 1]), pu, rightVectorBlock);
			}

			//ai = ||r||
			m_rsum.Set(calcCommon, right, m_r, right);
			m_rsum.ConjMul(calcCommon, right, m_rsum, right);
			T suma = m_rsum.ReduceSum(calcContract);
			ai.AddItem(suma);

			//ui = r / ai.
			suma = _Div(one, suma);
			rightVectorBlock.m_lstRange[0].SetAsElementIndex(i);
			*pu->Set(calcCommon, rightVectorBlock, m_r, rblock);
			*pu->Mul(calcCommon, rightVectorBlock, suma);

			if (i < uiTruncate - 1)
			{
				//p = Ad ui - aivi
				rightVectorIndex.m_lstRange[0].SetAsElementIndex(i);
				m_p.Contraction(calcContract, *pu, A, pblock, rightVectorIndex, left, right, right, TRUE);
				m_p.Axpy(calcCommon, pblock, ai[i], *pv, leftVectorBlock);

				//bi = || p ||
				m_psum.Set(calcCommon, left, m_p, left);
				m_psum.ConjMul(calcCommon, left, m_psum, left);
				bi.AddItem(m_psum.ReduceSum(calcContract));

				/// TODO: if bi = 0, early break
			}
		}

		//=============================================
		// 4 - finally, use ai and bi to create s-tensor
		ps->Zero(calcCommon);
		//set the diagnals
		UINT stride1[1];
		UINT stride2[1];
		UINT length[1];
		stride1[0] = uiTruncate + 1;
		stride2[0] = 1;
		length[0] = uiTruncate;
		ps->SetColumn(calcCommon, 1, 0, stride1, 0, stride2, length, uiTruncate, ai.GetData());

		//set bi
		length[0] = uiTruncate - 1;
		ps->SetColumn(calcCommon, 1, 1, stride1, 0, stride2, length, uiTruncate - 1, bi.GetData());

		return TRUE;
	}

protected:

	//========== used by Lanczos =============
	CNHostTensor<T> m_p;
	CNHostTensor<T> m_r;
	CNHostTensor<T> m_psum;
	CNHostTensor<T> m_rsum;

};

__END_NAMESPACE

#endif//#ifndef _CNMATRIXSVD_H_

//=============================================================================
// END OF FILE
//=============================================================================