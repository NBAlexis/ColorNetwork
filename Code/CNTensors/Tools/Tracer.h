//=============================================================================
// FILENAME : Tracer.h
// 
// DESCRIPTION:
// This is class for messages
//
// REVISION:
//  [01/06/2020 nbale]
//=============================================================================

#ifndef _TRACER_H_
#define _TRACER_H_

__BEGIN_NAMESPACE

__DEFINE_ENUM(EVerboseLevel,

    CRUCIAL,
    WARNING,
    GENERAL,
    DETAILED,
    PARANOIAC,

    ForceDWORD = 0x7fffffff,

    )

enum 
{
    _kTraceBuffSize = 4096,
};

class CNAPI CTracer
{
public:
    CTracer(void)
        : m_eLevel(CRUCIAL)
        , m_pStream(NULL)
        , m_pStdStream(NULL)
        , m_bLogDate(TRUE)
    {
        Initial(CRUCIAL);
    }

    ~CTracer(void)
    {
        if (NULL != m_pStream)
        {
            m_pStream->flush();
        }
        appSafeDelete(m_pStream);
    }

    inline void SetVerboseLevel(EVerboseLevel eLevel) { m_eLevel = eLevel; }

    inline void SetOutStream(const CCString& filename = _T("stdout"))
    {
        appSafeDelete(m_pStdStream);
        if (NULL != m_pStream)
        {
            m_pStream->flush();
            appSafeDelete(m_pStream);
        }

        m_pStdStream = new OSTREAM(COUT.rdbuf());
        UBOOL bShowHasFile = FALSE;
        if (filename == _T("stdout"))
        {
            m_pStream = NULL;
        }
        else if (filename == _T("timestamp"))
        {
            const CCString sRealFile = appStringFormat(_T("%d.log"), appGetTimeStamp());
            m_pStream = new OFSTREAM(sRealFile);
            bShowHasFile = TRUE;
        }
        else if (filename == _T("datetime"))
        {
            static TCHAR datetime[256];
            appGetTimeNow(datetime, 256);
            const CCString sRealFile = appStringFormat(_T("%s.log"), datetime);
            m_pStream = new OFSTREAM(sRealFile);
            bShowHasFile = TRUE;
        }
        else
        {
            m_pStream = new OFSTREAM(filename);
            bShowHasFile = TRUE;
        }

        if (NULL == m_pStdStream || (bShowHasFile && NULL == m_pStream))
        {
            printf(_T("ERROR: CTracer: no output stream."));
            if (NULL != m_pStream)
            {
                m_pStream->flush();
            }
            exit(EXIT_FAILURE);
        }
    }

    inline void Initial(EVerboseLevel eLevel = PARANOIAC, const CCString& filename = _T("stdout"))
    {
        m_eLevel = eLevel;
        m_pStdStream = new OSTREAM(COUT.rdbuf());
        UBOOL bShowHasFile = FALSE;
        if (filename == _T("stdout"))
        {
            m_pStream = NULL;
        }
        else if (filename == _T("timestamp"))
        {
            const CCString sRealFile = appStringFormat(_T("%d.log"), appGetTimeStamp());
            m_pStream = new OFSTREAM(sRealFile);
            bShowHasFile = TRUE;
        }
        else if (filename == _T("datetime"))
        {
            static TCHAR datetime[256];
            appGetTimeNow(datetime, 256);
            const CCString sRealFile = appStringFormat(_T("%s.log"), datetime);
            m_pStream = new OFSTREAM(sRealFile);
            bShowHasFile = TRUE;
        }
        else
        {
            m_pStream = new OFSTREAM(filename);
            bShowHasFile = TRUE;
        }

        if (NULL == m_pStdStream || (bShowHasFile && NULL == m_pStream))
        {
            printf(_T("ERROR: CTracer: no output stream."));
            if (NULL != m_pStream)
            {
                m_pStream->flush();
            }
            exit(EXIT_FAILURE);
        }
    }

    inline void Print(EVerboseLevel level, const TCHAR *format, va_list& arg)
    {
        if ((level <= m_eLevel))
        {
            //assert(NULL != m_pStdStream);
            if (NULL == m_pStdStream)
            {
                //Maybe the first initial is not entered?
            }

            if (CRUCIAL == level && NULL != m_pStdStream)
            {
                //red bold
                *m_pStdStream << _T("\033[31;1m");
            }
            else if (WARNING == level && NULL != m_pStdStream)
            {
                *m_pStdStream << _T("\033[35m");
            }
            else if (DETAILED == level && NULL != m_pStdStream)
            {
                //green
                *m_pStdStream << _T("\033[32m");
            }
            else if (PARANOIAC == level && NULL != m_pStdStream)
            {
                //dark yellow
                *m_pStdStream << _T("\033[33m");
            }

            if (m_bLogDate)
            {
                static TCHAR timeBuffer[256];
                if (level <= GENERAL)
                {
                    appGetTimeNow(timeBuffer, 256);
                    if (NULL != m_pStdStream)
                    {
                        *m_pStdStream << _T("[") << timeBuffer << "|" << m_sTraceHeader.c_str() << _T("]");
                    }
                    if (NULL != m_pStream)
                    {
                        *m_pStream << _T("[") << timeBuffer << "|" << m_sTraceHeader.c_str() << _T("]");
                    }
                }
            }

            appVsnprintf(m_cBuff, _kTraceBuffSize - 1, format, arg);
            if (NULL != m_pStdStream)
            {
                *m_pStdStream << m_cBuff;
            }
            
            if (NULL != m_pStream)
            {
                *m_pStream << m_cBuff;
#ifdef _CLG_DEBUG
                *m_pStream << std::flush;
#endif
            }

            if ((CRUCIAL == level || WARNING == level || PARANOIAC == level || DETAILED == level) && NULL != m_pStdStream)
            {
                *m_pStdStream << _T("\033[0m");
            }
        }
    }

    inline void Flush() const
    {
        if (NULL != m_pStream)
        {
            m_pStream->flush();
        }
    }

    inline void SetLogDate(UBOOL bLog)
    {
        m_bLogDate = bLog;
        m_bLogDateHist.RemoveAll();
    }
    inline void PushLogDate(UBOOL bLog)
    {
        m_bLogDateHist.AddItem(m_bLogDate);
        m_bLogDate = bLog;
    }
    inline void PopLogDate()
    {
        if (m_bLogDateHist.Num() > 0)
        {
            m_bLogDate = m_bLogDateHist.Pop();
        }
    }

    inline void SetLogHeader(const CCString& sHeader) { m_sTraceHeader = sHeader; }

private:

    EVerboseLevel m_eLevel;
    OSTREAM * m_pStream;
    OSTREAM * m_pStdStream;
    TCHAR m_cBuff[_kTraceBuffSize];
    UBOOL m_bLogDate;
    TArray<UBOOL> m_bLogDateHist;
    CCString m_sTraceHeader;
};

extern CNAPI void appInitialTracer(EVerboseLevel eLevel = PARANOIAC, const CCString& filename = _T("stdout"));
extern CNAPI void appVOut(EVerboseLevel eLevel, const TCHAR *format, ...);
extern CNAPI void _appCrucial(const TCHAR *format, ...);
extern CNAPI void _appWarning(const TCHAR* format, ...);
extern CNAPI void appGeneral(const TCHAR *format, ...);
extern CNAPI void appDetailed(const TCHAR *format, ...);
extern CNAPI void appParanoiac(const TCHAR *format, ...);

#define appAssert(exp) { if (!(exp)) { appCrucial(_T("assert failed %s\n"), _T(#exp)); } }
#define appCrucial(...) {TCHAR ___msg[1024];appSprintf(___msg, 1024, __VA_ARGS__);_appCrucial(_T("%s(%d): Error: %s\n"), _T(__FILE__), __LINE__, ___msg);}
#define appWarning(...) {TCHAR ___msg[1024];appSprintf(___msg, 1024, __VA_ARGS__);_appWarning(_T("%s(%d): Warning: %s\n"), _T(__FILE__), __LINE__, ___msg);}

extern CNAPI CTracer GTracer;

extern CNAPI void CNSetupLog(class CParameters& param);

inline void appSetTracer(EVerboseLevel eLevel, const CCString& filename)
{
    GTracer.SetVerboseLevel(eLevel);
    GTracer.SetOutStream(filename);
}

inline void appFlushLog()
{
    GTracer.Flush();
}

inline void appSetLogDate(UBOOL bLog)
{
    GTracer.SetLogDate(bLog);
}

inline void appPushLogDate(UBOOL bLog)
{
    GTracer.PushLogDate(bLog);
}

inline void appPopLogDate()
{
    GTracer.PopLogDate();
}

inline void appSetLogHeader(const CCString& sHeader)
{
    GTracer.SetLogHeader(sHeader);
}

__END_NAMESPACE

#endif //_TRACER_H_

//=============================================================================
// END OF FILE
//=============================================================================
