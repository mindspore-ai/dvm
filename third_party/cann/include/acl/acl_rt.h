/**
 * @file acl_rt.h
 *
 * Minimal ACL runtime declarations used by this project.
 */
#ifndef INC_EXTERNAL_ACL_ACL_RT_H_
#define INC_EXTERNAL_ACL_ACL_RT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

typedef int aclError;
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtDrvMemHandle;

static const int ACL_ERROR_NONE = 0;
static const int ACL_SUCCESS = 0;

#define ACL_EVENT_SYNC 0x00000001u
#define ACL_EVENT_CAPTURE_STREAM_PROGRESS 0x00000002u
#define ACL_EVENT_TIME_LINE 0x00000008u

typedef enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST,
  ACL_MEMCPY_HOST_TO_DEVICE,
  ACL_MEMCPY_DEVICE_TO_HOST,
  ACL_MEMCPY_DEVICE_TO_DEVICE,
  ACL_MEMCPY_DEFAULT,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
  ACL_MEM_MALLOC_HUGE_FIRST,
  ACL_MEM_MALLOC_HUGE_ONLY,
  ACL_MEM_MALLOC_NORMAL_ONLY,
  ACL_MEM_MALLOC_HUGE_FIRST_P2P,
  ACL_MEM_MALLOC_HUGE_ONLY_P2P,
  ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
  ACL_MEM_TYPE_LOW_BAND_WIDTH = 0x0100,
  ACL_MEM_TYPE_HIGH_BAND_WIDTH = 0x1000,
} aclrtMemMallocPolicy;

typedef enum aclrtMemAttr {
  ACL_DDR_MEM,
  ACL_HBM_MEM,
  ACL_DDR_MEM_HUGE,
  ACL_DDR_MEM_NORMAL,
  ACL_HBM_MEM_HUGE,
  ACL_HBM_MEM_NORMAL,
  ACL_DDR_MEM_P2P_HUGE,
  ACL_DDR_MEM_P2P_NORMAL,
  ACL_HBM_MEM_P2P_HUGE,
  ACL_HBM_MEM_P2P_NORMAL,
} aclrtMemAttr;

typedef enum aclrtMemLocationType {
  ACL_MEM_LOCATION_TYPE_HOST = 0,
  ACL_MEM_LOCATION_TYPE_DEVICE,
} aclrtMemLocationType;

typedef struct aclrtMemLocation {
  uint32_t id;
  aclrtMemLocationType type;
} aclrtMemLocation;

typedef enum aclrtMemAllocationType {
  ACL_MEM_ALLOCATION_TYPE_PINNED = 0,
} aclrtMemAllocationType;

typedef enum aclrtMemHandleType {
  ACL_MEM_HANDLE_TYPE_NONE = 0,
} aclrtMemHandleType;

typedef struct aclrtPhysicalMemProp {
  aclrtMemHandleType handleType;
  aclrtMemAllocationType allocationType;
  aclrtMemAttr memAttr;
  aclrtMemLocation location;
  uint64_t reserve;
} aclrtPhysicalMemProp;

ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId);
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId);
ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId);
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count);
ACL_FUNC_VISIBILITY const char *aclrtGetSocName(void);

ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr);
ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size);
ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr);

ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count,
                                         aclrtMemcpyKind kind);
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                                              aclrtMemcpyKind kind, aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);

ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream);
ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStream(aclrtStream stream);

ACL_FUNC_VISIBILITY aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag);
ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event);
ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent);

ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment,
                                                    void *expectPtr, uint64_t flags);
ACL_FUNC_VISIBILITY aclError aclrtReleaseMemAddress(void *virPtr);
ACL_FUNC_VISIBILITY aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size,
                                                 const aclrtPhysicalMemProp *prop, uint64_t flags);
ACL_FUNC_VISIBILITY aclError aclrtFreePhysical(aclrtDrvMemHandle handle);
ACL_FUNC_VISIBILITY aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle,
                                         uint64_t flags);
ACL_FUNC_VISIBILITY aclError aclrtUnmapMem(void *virPtr);

ACL_FUNC_VISIBILITY aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType handleType,
                                                             uint64_t flags, uint64_t *shareableHandle);
ACL_FUNC_VISIBILITY aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId,
                                                               aclrtDrvMemHandle *handle);
ACL_FUNC_VISIBILITY aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t *pid, size_t pidNum);

ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId);
ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetBareTgid(int32_t *pid);

#ifdef __cplusplus
}
#endif

#endif  // INC_EXTERNAL_ACL_ACL_RT_H_
