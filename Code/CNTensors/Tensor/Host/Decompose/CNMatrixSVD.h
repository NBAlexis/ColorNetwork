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

class CNAPI CNMatrixSVD
{
public:
	CNMatrixSVD()
	{

	}

	/**
	* use Lanzcos method to calculate
	* t1[left, right] = v[left,mid1] s[mid1,mid2] u[mid2, right], where order(mid1)=order(mid2)=truncate
	*/
	template<class T>
	UBOOL DecomposeLanzcos(const CNHostTensor<T>& t1, const CNIndexBlock& left, const CNIndexBlock& right, CNHostTensor<T>** v, CNHostTensor<T>** s, CNHostTensor<T>** u, UINT uiTruncate);
};

__END_NAMESPACE

#endif//#ifndef _CNMATRIXSVD_H_

//=============================================================================
// END OF FILE
//=============================================================================