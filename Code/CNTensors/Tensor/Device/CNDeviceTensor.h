//=============================================================================
// FILENAME : CNDeviceTensor.h
// 
// DESCRIPTION:
// 
//
// REVISION:
//  [18/06/2021 nbalexis]
//=============================================================================
#ifndef _CNDEVICETENSOR_H_
#define _CNDEVICETENSOR_H_

__BEGIN_NAMESPACE

/**
 * A 32 order tensor with all dim=2 needs 64G memory
 * Any tensor larger than this is not capable
 */
constexpr BYTE _kMaxSupportedOrder = 32;

class CNAPI CNDeviceTensorPlaceHolder
{
public:
    virtual ~CNDeviceTensorPlaceHolder() {}
};

/**
* It is designed to work like this:
*
* device_tensor1 = host_tensor1.GetDeviceTensor()
* device_tensor2 = host_tensor2.GetDeviceTensor()
* device_tensor3 = contractor.contract(device_tensor1, device_tensor2, other_parameters)
* host_tensor3 = create_host_tensor(device_tensor3)
*
*/
template<class T>
class __DLL_EXPORT CNDeviceTensor : public CNDeviceTensorPlaceHolder
{
public:

    T* m_pDeviceDataBuffer;
    UINT* m_pDeviceStrides;
    UINT* m_pDeviceLength;
    UINT m_iDim;
};

__END_NAMESPACE

#endif//#ifndef _CNDEVICETENSOR_H_

//=============================================================================
// END OF FILE
//=============================================================================