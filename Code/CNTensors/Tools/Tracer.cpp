//=============================================================================
// FILENAME : Tracer.cpp
// 
// DESCRIPTION:
// This is class for messages
//
// REVISION:
//  [01/06/2020 nbale]
//=============================================================================

#include "CNTensorsPch.h"

__BEGIN_NAMESPACE

CNAPI CTracer GTracer;

void CNSetupLog(CParameters& params)
{
    //Setup outputs
    CCString verboselevel;
    EVerboseLevel eVerbLevel = CRUCIAL;
    CCString sVerbFile = _T("stdout");
    const UBOOL fetchVerbLevel = params.FetchStringValue(_T("VerboseLevel"), verboselevel);
    const UBOOL fetchVerbFile = params.FetchStringValue(_T("VerboseOutput"), sVerbFile);
    if (fetchVerbLevel || fetchVerbFile) //do NOT put fetch string in if, it will enter if when the first is TRUE
    {
        eVerbLevel = __STRING_TO_ENUM(EVerboseLevel, verboselevel);
        appSetTracer(eVerbLevel, sVerbFile);
    }

    //check whether to log parameter file
    INT iTag = 0;
    //appGeneral(_T("============================== Parameter =============================\n\n"));
    __CheckTag(_T("ShowParameterContent"), params.Dump());
    //appGeneral(_T("============================== GPU =============================\n\n"));
    __CheckTag(_T("ShowDeviceInformation"), CCudaHelper::DeviceQuery());

    appGeneral(_T("============================== Log Start =============================\n\n"));
}

/**
*
*
*/
CNAPI void appInitialTracer(EVerboseLevel eLevel, const CCString& filename)
{
    GTracer.Initial(eLevel, filename);
}

/**
*
*
*/
CNAPI void appVOut(EVerboseLevel level, const TCHAR *format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GTracer.Print(level, format, arg);
        va_end(arg);
    }
}

/**
*
*
*/
CNAPI void _appCrucial(const TCHAR *format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GTracer.Print(CRUCIAL, format, arg);
        GTracer.Flush();
        va_end(arg);
    }

    _CNBREAK;
}

CNAPI void _appWarning(const TCHAR* format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GTracer.Print(WARNING, format, arg);
        GTracer.Flush();
        va_end(arg);
    }
}

/**
*
*
*/
CNAPI void appGeneral(const TCHAR *format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GTracer.Print(GENERAL, format, arg);
        va_end(arg);
    }
}

/**
*
*
*/
CNAPI void appDetailed(const TCHAR *format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GTracer.Print(DETAILED, format, arg);
        va_end(arg);
    }
}

/**
*
*
*/
CNAPI void appParanoiac(const TCHAR *format, ...)
{
    va_list arg;
    {
        va_start(arg, format);
        GTracer.Print(PARANOIAC, format, arg);
        va_end(arg);
    }
}

__END_NAMESPACE

//====================================================================
//====================================================================
