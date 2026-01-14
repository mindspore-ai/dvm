/**
 * @file prof_api.h
 *
 * Minimal profiling API declarations used by this project.
 */
#ifndef PROF_API_H
#define PROF_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

#define MSPROF_DATA_HEAD_MAGIC_NUM 0x5A5AU
#define MSPROF_REPORT_DATA_MAGIC_NUM 0x5A5AU

typedef void *VOID_PTR;

/* Report levels/types used by dvm */
#define MSPROF_REPORT_NODE_LEVEL 10000U
#define MSPROF_REPORT_NODE_BASIC_INFO_TYPE 0U
#define MSPROF_REPORT_NODE_TENSOR_INFO_TYPE 1U
#define MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE 4U
#define MSPROF_REPORT_NODE_LAUNCH_TYPE 5U

/* Tensor info constants */
#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8
#define MSPROF_GE_TENSOR_DATA_NUM 5

enum MsprofGeTensorType {
    MSPROF_GE_TENSOR_TYPE_INPUT = 0,
    MSPROF_GE_TENSOR_TYPE_OUTPUT,
};

/* DataTypeConfig MASK (minimal) */
#define PROF_TASK_TIME_L1_MASK 0x00000002ULL
#define PROF_TASK_TIME_MASK 0x00000800ULL

#define MSPROF_COMPACT_INFO_DATA_LENGTH 40
#define MSPROF_ADDTIONAL_INFO_DATA_LENGTH 232

#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095

typedef int32_t (*ProfCommandHandle)(uint32_t type, VOID_PTR data, uint32_t len);

/* profiling control type */
enum ProfCtrlType {
    PROF_CTRL_INVALID = 0,
    PROF_CTRL_SWITCH,
    PROF_CTRL_REPORTER,
    PROF_CTRL_STEPINFO,
    PROF_CTRL_BUTT
};

enum MsprofCommandHandleType {
    PROF_COMMANDHANDLE_TYPE_INIT = 0,
    PROF_COMMANDHANDLE_TYPE_START,
    PROF_COMMANDHANDLE_TYPE_STOP,
    PROF_COMMANDHANDLE_TYPE_FINALIZE,
    PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE,
    PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE,
    PROF_COMMANDHANDLE_TYPE_MAX
};

enum MsprofErrorCode {
    MSPROF_ERROR_NONE = 0,
    MSPROF_ERROR_MEM_NOT_ENOUGH,
    MSPROF_ERROR_GET_ENV,
    MSPROF_ERROR_CONFIG_INVALID,
    MSPROF_ERROR_ACL_JSON_OFF,
    MSPROF_ERROR,
    MSPROF_ERROR_UNINITIALIZE,
};

/* profiling command info */
#define MSPROF_MAX_DEV_NUM 64
struct MsprofCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;  // MB
    uint32_t profDataLen;
    char path[PATH_LEN_MAX + 1];
    char profData[PARAM_LEN_MAX + 1];
};

struct MsprofCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[MSPROF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
    uint32_t cacheFlag;
    struct MsprofCommandHandleParams params;
};

#pragma pack(1)
struct MsprofNodeBasicInfo {
    uint64_t opName;
    uint32_t taskType;
    uint64_t opType;
    uint32_t blockDim;
    uint32_t opFlag;
};

struct MsrofTensorData {
    uint32_t tensorType;
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};

struct MsprofTensorInfo {
    uint64_t opName;
    uint32_t tensorNum;
    struct MsrofTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
};

#define MSPROF_CTX_ID_MAX_NUM 55
struct MsprofContextIdInfo {
    uint64_t opName;
    uint32_t ctxIdNum;
    uint32_t ctxIds[MSPROF_CTX_ID_MAX_NUM];
};
#pragma pack()

struct MsprofApi {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;
};

struct MsprofCompactInfo {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    union {
        uint8_t info[MSPROF_COMPACT_INFO_DATA_LENGTH];
        struct MsprofNodeBasicInfo nodeBasicInfo;
    } data;
};

struct MsprofAdditionalInfo {
#ifdef __cplusplus
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
#else
    uint16_t magicNumber;
#endif
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t data[MSPROF_ADDTIONAL_INFO_DATA_LENGTH];
};

MSVP_PROF_API uint64_t MsprofSysCycleTime();
MSVP_PROF_API uint64_t MsprofGetHashId(const char *hashInfo, size_t length);
MSVP_PROF_API int32_t MsprofReportApi(uint32_t agingFlag, const struct MsprofApi *api);
MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
MSVP_PROF_API int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
MSVP_PROF_API int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);

#ifdef __cplusplus
}
#endif

#endif  // PROF_API_H
