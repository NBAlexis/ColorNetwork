//=============================================================================
// FILENAME : CFile.h
// 
// DESCRIPTION:
// 
// REVISION[d-m-y]:
//  [23/07/2022 nbale]
//=============================================================================
#pragma once

#ifndef _CFILE_H_
#define _CFILE_H_

//#pragma TODO (use boost::filesystem)


__BEGIN_NAMESPACE

enum { DETAILED_STATS = 0 };
enum { kMaxPathNameLength = 1024 };
enum { kInvalidReturn = -1 };

class CNAPI CFile
{
public:
    void Init() {}

    /**
    * Need to free the pointer
    */
    static BYTE* ReadAllBytes(const TCHAR* sFilename, UINT& size);
    static UBOOL WriteAllBytes(const TCHAR* sFilename, BYTE* data, UINT uiSize);
    static CCString ReadAllText(const TCHAR* sFilename);
    static UBOOL WriteAllText(const TCHAR* sFilename, const CCString& data);

    static UBOOL IsFileExist(const CCString& sFileName);
    static UBOOL AppendAllText(const TCHAR* sFilename, const CCString& data);

    static OFSTREAM* OpenFileToLoad(const CCString& sFilename, UBOOL bBinary = TRUE);
    static OFSTREAM* OpenFileToSave(const CCString& sFilename, UBOOL bBinary = TRUE, UBOOL bAppend = FALSE);

};

__END_NAMESPACE

#endif //#ifndef _CFILE_H_

//=============================================================================
// END OF FILE
//=============================================================================

