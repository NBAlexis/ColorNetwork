//=============================================================================
// FILENAME : CNIndex.h
// 
// DESCRIPTION:
// This is index class
//   It records an ID, a dimension, a name, (and maybe several tags?)
//
// For Quantum number symmetry we may write another class?
//
//
//  For index {a < b < c < d}, it is always ((a * db + b) * dc + c) * dd + d
//
// REVISION:
//  [04/30/2020 nbalexis]
//=============================================================================

#ifndef _CNINDEX_H_
#define _CNINDEX_H_

__BEGIN_NAMESPACE


class CNAPI CNDeviceTensorIndex
{
public:
    __device__ CNDeviceTensorIndex(UINT order, const UINT* dims)
    {
        
    }
    

protected:


};

__END_NAMESPACE

#endif //#ifndef _CNINDEX_H_

//=============================================================================
// END OF FILE
//=============================================================================
