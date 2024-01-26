/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: kernel.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_KERNEL_H
#define CCE_RUNTIME_KERNEL_H

#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

// If you need export the function of this library in Win32 dll, use __declspec(dllexport)
#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __declspec(dllexport)
#else
#define RTS_API
#endif
#endif

#ifndef char_t
typedef char char_t;
#endif

typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0; // success

/**
 * @ingroup dvrt_base
 * @brief stream handle.
 */
typedef void *rtStream_t;

/**
 * @ingroup dvrt_base
 * @brief Args handle.
 */
typedef void *rtLaunchArgsHandle;

/**
 * @ingroup dvrt_base
 * @brief Program handle.
 */
typedef void *rtBinHandle;

/**
 * @ingroup dvrt_base
 * @brief Kernel handle.
 */
typedef void *rtFuncHandle;

/**
 * @ingroup rt_kernel
 * @brief shared memory data control
 */
typedef struct tagRtSmData {
    uint64_t L2_mirror_addr;          // preload or swap source addr
    uint32_t L2_data_section_size;    // every data size
    uint8_t L2_preload;               // 1 - preload from mirrorAddr, 0 - no preload
    uint8_t modified;                 // 1 - data will be modified by kernel, 0 - no modified
    uint8_t priority;                 // data priority
    int8_t prev_L2_page_offset_base;  // remap source section offset
    uint8_t L2_page_offset_base;      // remap destination section offset
    uint8_t L2_load_to_ddr;           // 1 - need load out, 0 - no need
    uint8_t reserved[2];              // reserved
} rtSmData_t;

/**
 * @ingroup rt_kernel
 * @brief device binary type
 */
typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;

/**
 * @ingroup rt_kernel
 * @brief shared memory description
 */
typedef struct tagRtSmCtrl {
    rtSmData_t data[8];  // data description
    uint64_t size;       // max page Num
    uint8_t remap[64];   /* just using for static remap mode, default:0xFF
                          array index: virtual l2 page id, array value: physic l2 page id */
    uint8_t l2_in_main;  // 0-DDR, 1-L2, default:0xFF
    uint8_t reserved[3];
} rtSmDesc_t;

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_PLAIN 0xabceed50U

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AICPU 0xabceed51U

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AIVEC 0xabceed52U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_ELF_AICPU 0x41415243U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicube
 */
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343U

/**
 * @ingroup rt_kernel_flags
 * @brief kernel op bit flags
 */
#define RT_KERNEL_DEFAULT (0x00U)
#define RT_KERNEL_CONVERT (0x01U)
#define RT_KERNEL_DUMPFLAG (0x02U)
#define RT_FUSION_KERNEL_DUMPFLAG (0x04U)
#define RT_KERNEL_CUSTOM_AICPU (0x08U)
#define RT_KERNEL_FFTSPLUS_DYNAMIC_SHAPE_DUMPFLAG (0x10U)
#define RT_KERNEL_FFTSPLUS_STATIC_SHAPE_DUMPFLAG  (0x20U)

// STARS topic scheduler sqe : topic_type
#define RT_KERNEL_DEVICE_FIRST (0x10U)
#define RT_KERNEL_HOST_ONLY (0x20U)
#define RT_KERNEL_HOST_FIRST (0x40U)
#define RT_KERNEL_BIUPERF_FLAG (0x80U)
#define RT_KERNEL_CMDLIST_NOT_FREE                (0x40U) // cmdlist does not need to be released by the runtime.

/**
 * @ingroup rt_kernel
 * @brief kernel mode
**/
#define RT_DEFAULT_KERNEL_MODE (0x00U)
#define RT_NORMAL_KERNEL_MODE (0x01U)
#define RT_ALL_KERNEL_MODE (0x02U)

/**
 * @ingroup rt_kernel
 * @brief SHAPE kernel type
**/
#define RT_STATIC_SHAPE_KERNEL (0x00U)
#define RT_DYNAMIC_SHAPE_KERNEL (0x01U)

/**
 * @ingroup rt_kernel
 * @brief kernel L1 Fusion Dump bit flags
 */
#define RT_DDR_ADDR (0x0U)

/**
 * @ingroup rt_kernel
 * @brief register device binary
 * @param [in] bin   device binary description
 * @param [out] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);

/**
 * @ingroup rt_kernel
 * @brief register fast memeory device binary
 * @param [in] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtBinaryRegisterToFastMemory(void *hdl);

/**
 * @ingroup rt_kernel
 * @brief unregister device binary
 * @param [in] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDevBinaryUnRegister(void *hdl);

/**
 * @ingroup rt_kernel
 * @brief register device binary metadata
 * @param [in] hdl    device binary description
 * @param [in] metadata  device binary metadata
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtMetadataRegister(void *hdl, const char_t *metadata);

/**
 * @ingroup rt_kernel
 * @brief register device binary dependency
 * @param [in] mHandle   master device binary description
 * @param [in] sHandle   slave device binary description
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDependencyRegister(void *mHandle, void *sHandle);

/**
 * @ingroup rt_kernel
 * @brief register device function
 * @param [in] binHandle   device binary handle
 * @param [in] stubFunc   stub function
 * @param [in] stubName   stub function name
 * @param [in] kernelInfoExt   kernel Info extension. device function description or tiling key,
 *                             depending static shape or dynmaic shape.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                                     const void *kernelInfoExt, uint32_t funcMode);

/**
 * @ingroup rt_kernel
 * @brief find stub function by name
 * @param [in] stubName   stub function name
 * @param [out] stubFunc   stub function
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetFunctionByName(const char_t *stubName, void **stubFunc);

/**
 * @ingroup rt_kernel
 * @brief find addr by stub func
 * @param [in] stubFunc   stub function
 * @param [out] addr
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetAddrByFun(const void *stubFunc, void **addr);
/**
 * @ingroup rt_kernel
 * @brief query registered or not by stubName
 * @param [in] stubName   stub function name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtQueryFunctionRegistered(const char_t *stubName);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stm);
/**
 * @ingroup rt_kernel(abandoned)
 * @brief launch kernel to device
 * @param [in] args       argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] flags      launch flags
 * @param [in] stm     associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchEx(void *args, uint32_t argsSize, uint32_t flags, rtStream_t stm);

/**
 * @ingroup rt_kernel(in use)
 * @brief launch kernel to device
 * @param [in] opName     opkernel name
 * @param [in] args       argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] flags      launch flags
 * @param [in] stm     associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchFwk(const char_t *opName, void *args, uint32_t argsSize, uint32_t flags,
                                    rtStream_t rtStream);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device with previous setting kernel argment
 *        and call argment
 * @param [in] stubFunc   stub function
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLaunch(const void *stubFunc);

/**
 * @ingroup rt_kernel
 * @brief Create Args Handle.
 * @param [in] argsSize   args Size
 * @param [in] hostInfoTotalSize   hostInfoTotal Size
 * @param [in] hostInfoNum   hostInfo num
 * @param [in] argsData   args Data
 * @param [out] argsHandle   args Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtCreateLaunchArgs(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum,
                             void* argsData, rtLaunchArgsHandle* argsHandle);

/**
 * @ingroup rt_kernel
 * @brief Destroy Args Handle.
 * @param [in] argsHandle   args Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtDestroyLaunchArgs(rtLaunchArgsHandle argsHandle);

/**
 * @ingroup rt_kernel
 * @brief Reset Args Handle Info.
 * @param [in] argsHandle   args Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtResetLaunchArgs(rtLaunchArgsHandle argsHandle);

/**
 * @ingroup rt_kernel
 * @brief Append address info to Args Handle.
 * @param [in] argsHandle   args Handle
 * @param [in] addrInfo   address info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtAppendLaunchAddrInfo(rtLaunchArgsHandle argsHandle, void *addrInfo);

/**
 * @ingroup rt_kernel
 * @brief Append Host info to args  Handle.
 * @param [in] argsHandle   args Handle
 * @param [in] hostInfoSize   host Info Size
 * @param [out] hostInfo host info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtAppendLaunchHostInfo(rtLaunchArgsHandle argsHandle, size_t hostInfoSize, void **hostInfo);

/**
 * @ingroup rt_kernel
 * @brief Registers and parses the bin file and loads it to the device.
 * @param [in] bin   device binary description
 * @param [out] binHandle   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtBinaryLoad(const rtDevBinary_t *bin, rtBinHandle *binHandle);

/**
 * @ingroup rt_kernel
 * @brief Find funcHandle based on binHandle and tilingKey.
 * @param [in] binHandle  funcHandle
  * @param [in] tilingKey   tilingKey
 * @param [out] funcHandle   funcHandle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtBinaryGetFunction(const rtBinHandle binHandle, const uint64_t tilingKey, rtFuncHandle *funcHandle);

/**
 * @ingroup rt_kernel
 * @brief UnLoad binary
 * @param [in] binHandle  Binary Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtBinaryUnLoad(rtBinHandle binHandle);

/**
 * @ingroup rt_kernel
 * @brief Kernel Launch to device
 * @param [in] funcHandle  function Handle
 * @param [in] blockDim  block dimentions
 * @param [in] argsHandle  args Handle
 * @param [in] stm  associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtLaunchKernelByFuncHandle(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                     rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief get Saturation Status task
 * @param [in] outputAddrPtr  pointer to op output addr
 * @param [in] outputSize   op output size
 * @param [in] stm  associated stream
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtGetDeviceSatStatus(void * const outputAddrPtr, const uint64_t outputSize, rtStream_t stm);
#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_KERNEL_H

