//=============================================================================
// FILENAME : COpenCLStart.cpp
// 
// DESCRIPTION:
//
//
// REVISION:
//  [31/08/2021 nbale]
//=============================================================================
#include "CNTensorsPch.h"
#include "cl/opencl.h"

__BEGIN_NAMESPACE

COpenCLStart::COpenCLStart()
{

}

COpenCLStart::~COpenCLStart()
{

}

void COpenCLStart::Initial(const class CParameters& sConfigFile)
{
    
}

void COpenCLStart::Exit()
{
    
}

void COpenCLStart::PrintDeviceInfo()
{
    bool bPassed = true;
    std::string sProfileString = "oclDeviceQuery, Platform Name = ";

    // Get OpenCL platform ID for NVIDIA if avaiable, otherwise default
    appGeneral("OpenCL SW Info:\n\n");
    char cBuffer[1024];
    cl_platform_id clSelectedPlatformID = NULL;
    char chBuffer[1024];
    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
        appGeneral(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
    }
    else
    {
        if (num_platforms == 0)
        {
            appGeneral("No OpenCL platform found!\n\n");
        }
        else
        {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                appGeneral("Failed to allocate memory for cl_platform ID's!\n\n");
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
            for (cl_uint i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if (ciErrNum == CL_SUCCESS)
                {
                    appGeneral(_T("platform :%s"), chBuffer);
                    if (strstr(chBuffer, "NVIDIA") != NULL)
                    {
                        clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if (clSelectedPlatformID == NULL)
            {
                appGeneral("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }

    /*
    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo(clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_NAME: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    }
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", Platform Version = ";

    ciErrNum = clGetPlatformInfo(clSelectedPlatformID, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_VERSION: \t%s\n", cBuffer);
        sProfileString += cBuffer;
    }
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    sProfileString += ", SDK Revision = ";

    // Log OpenCL SDK Revision # 
    shrLog(" OpenCL SDK Revision: \t%s\n\n\n", OCL_SDKREVISION);
    sProfileString += OCL_SDKREVISION;
    sProfileString += ", NumDevs = ";

    // Get and log OpenCL device info 
    cl_uint ciDeviceCount;
    cl_device_id* devices;
    shrLog("OpenCL Device Info:\n\n");
    ciErrNum = clGetDeviceIDs(clSelectedPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);

    // check for 0 devices found or errors... 
    if (ciDeviceCount == 0)
    {
        shrLog(" No devices found supporting OpenCL (return code %i)\n\n", ciErrNum);
        bPassed = false;
        sProfileString += "0";
    }
    else if (ciErrNum != CL_SUCCESS)
    {
        shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
        bPassed = false;
    }
    else
    {
        // Get and log the OpenCL device ID's
        shrLog(" %u devices found supporting OpenCL:\n\n", ciDeviceCount);
        char cTemp[2];
#ifdef WIN32
        sprintf_s(cTemp, 2 * sizeof(char), "%u", ciDeviceCount);
#else
        sprintf(cTemp, "%u", ciDeviceCount);
#endif
        sProfileString += cTemp;
        if ((devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ciDeviceCount)) == NULL)
        {
            shrLog(" Failed to allocate memory for devices !!!\n\n");
            bPassed = false;
        }
        ciErrNum = clGetDeviceIDs(clSelectedPlatformID, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);
        if (ciErrNum == CL_SUCCESS)
        {
            //Create a context for the devices
            cl_context cxGPUContext = clCreateContext(0, ciDeviceCount, devices, NULL, NULL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog("Error %i in clCreateContext call !!!\n\n", ciErrNum);
                bPassed = false;
            }
            else
            {
                // show info for each device in the context
                for (unsigned int i = 0; i < ciDeviceCount; ++i)
                {
                    shrLog(" ---------------------------------\n");
                    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
                    shrLog(" Device %s\n", cBuffer);
                    shrLog(" ---------------------------------\n");
                    oclPrintDevInfo(LOGBOTH, devices[i]);
                    sProfileString += ", Device = ";
                    sProfileString += cBuffer;
                }

                // Determine and show image format support 
                cl_uint uiNumSupportedFormats = 0;

                // 2D
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY,
                    CL_MEM_OBJECT_IMAGE2D,
                    NULL, NULL, &uiNumSupportedFormats);
                cl_image_format* ImageFormats = new cl_image_format[uiNumSupportedFormats];
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY,
                    CL_MEM_OBJECT_IMAGE2D,
                    uiNumSupportedFormats, ImageFormats, NULL);
                shrLog("  ---------------------------------\n");
                shrLog("  2D Image Formats Supported (%u)\n", uiNumSupportedFormats);
                shrLog("  ---------------------------------\n");
                shrLog("  %-6s%-16s%-22s\n\n", "#", "Channel Order", "Channel Type");
                for (unsigned int i = 0; i < uiNumSupportedFormats; i++)
                {
                    shrLog("  %-6u%-16s%-22s\n", (i + 1),
                        oclImageFormatString(ImageFormats[i].image_channel_order),
                        oclImageFormatString(ImageFormats[i].image_channel_data_type));
                }
                shrLog("\n");
                delete[] ImageFormats;

                // 3D
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY,
                    CL_MEM_OBJECT_IMAGE3D,
                    NULL, NULL, &uiNumSupportedFormats);
                ImageFormats = new cl_image_format[uiNumSupportedFormats];
                clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY,
                    CL_MEM_OBJECT_IMAGE3D,
                    uiNumSupportedFormats, ImageFormats, NULL);
                shrLog("  ---------------------------------\n");
                shrLog("  3D Image Formats Supported (%u)\n", uiNumSupportedFormats);
                shrLog("  ---------------------------------\n");
                shrLog("  %-6s%-16s%-22s\n\n", "#", "Channel Order", "Channel Type");
                for (unsigned int i = 0; i < uiNumSupportedFormats; i++)
                {
                    shrLog("  %-6u%-16s%-22s\n", (i + 1),
                        oclImageFormatString(ImageFormats[i].image_channel_order),
                        oclImageFormatString(ImageFormats[i].image_channel_data_type));
                }
                shrLog("\n");
                delete[] ImageFormats;
            }
        }
        else
        {
            shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            bPassed = false;
        }
    }

    // masterlog info
    sProfileString += "\n";
    shrLogEx(LOGBOTH | MASTER, 0, sProfileString.c_str());

    // Log system info(for convenience:  not specific to OpenCL) 
    shrLog("\nSystem Info: \n\n");
#ifdef _WIN32
    SYSTEM_INFO stProcInfo;         // processor info struct
    OSVERSIONINFO stOSVerInfo;      // Win OS info struct
    SYSTEMTIME stLocalDateTime;     // local date / time struct 

    // processor
    SecureZeroMemory(&stProcInfo, sizeof(SYSTEM_INFO));
    GetSystemInfo(&stProcInfo);

    // OS
    SecureZeroMemory(&stOSVerInfo, sizeof(OSVERSIONINFO));
    stOSVerInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    GetVersionEx(&stOSVerInfo);

    // date and time
    GetLocalTime(&stLocalDateTime);

    // write time and date to logs
    shrLog(" Local Time/Date = %i:%i:%i, %i/%i/%i\n",
        stLocalDateTime.wHour, stLocalDateTime.wMinute, stLocalDateTime.wSecond,
        stLocalDateTime.wMonth, stLocalDateTime.wDay, stLocalDateTime.wYear);

    // write proc and OS info to logs
    shrLog(" CPU Arch: %i\n CPU Level: %i\n # of CPU processors: %u\n Windows Build: %u\n Windows Ver: %u.%u %s\n\n\n",
        stProcInfo.wProcessorArchitecture, stProcInfo.wProcessorLevel, stProcInfo.dwNumberOfProcessors,
        stOSVerInfo.dwBuildNumber, stOSVerInfo.dwMajorVersion, stOSVerInfo.dwMinorVersion,
        (stOSVerInfo.dwMajorVersion >= 6) ? "(Windows Vista / Windows 7)" : "");
#endif

#ifdef MAC
#else
#ifdef UNIX
    char timestr[255];
    time_t now = time(NULL);
    struct tm* ts;

    ts = localtime(&now);

    strftime(timestr, 255, " %H:%M:%S, %m/%d/%Y", ts);

    // write time and date to logs
    shrLog(" Local Time/Date = %s\n", timestr);

    // write proc and OS info to logs

    // parse /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo"); // open the file in /proc        
    std::string tmp;

    int cpu_num = 0;
    std::string cpu_name = "none";

    do {
        cpuinfo >> tmp;

        if (tmp == "processor")
            cpu_num++;

        if (tmp == "name")
        {
            cpuinfo >> tmp; // skip :

            std::stringstream tmp_stream("");
            do {
                cpuinfo >> tmp;
                if (tmp != std::string("stepping")) {
                    tmp_stream << tmp.c_str() << " ";
                }
            } while (tmp != std::string("stepping"));

            cpu_name = tmp_stream.str();
        }

    } while ((!cpuinfo.eof()));

    // Linux version
    std::ifstream version("/proc/version");
    char versionstr[255];

    version.getline(versionstr, 255);

    shrLog(" CPU Name: %s\n # of CPU processors: %u\n %s\n\n\n",
        cpu_name.c_str(), cpu_num, versionstr);
#endif
#endif
*/
}


__END_NAMESPACE



//=============================================================================
// END OF FILE
//=============================================================================
