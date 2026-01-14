/**
 * @file acl_prof.h
 *
 * Minimal profiling API declarations used by this project.
 */
#ifndef INC_EXTERNAL_ACL_PROF_H_
#define INC_EXTERNAL_ACL_PROF_H_

#include <stddef.h>
#include <stdint.h>
#include "acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MSVP_PROF_API
#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif
#endif

#define ACL_PROF_ACL_API        0x0001ULL
#define ACL_PROF_TASK_TIME      0x0002ULL
#define ACL_PROF_AICORE_METRICS 0x0004ULL
#define ACL_PROF_AICPU          0x0008ULL
#define ACL_PROF_TRAINING_TRACE 0x0040ULL

typedef enum {
    ACL_AICORE_ARITHMETIC_UTILIZATION = 0,
} aclprofAicoreMetrics;

typedef struct aclprofConfig aclprofConfig;
typedef struct aclprofAicoreEvents aclprofAicoreEvents;

MSVP_PROF_API aclError aclprofInit(const char *profilerResultPath, size_t length);
MSVP_PROF_API aclError aclprofFinalize();
MSVP_PROF_API aclError aclprofStart(const aclprofConfig *profilerConfig);
MSVP_PROF_API aclprofConfig *aclprofCreateConfig(uint32_t *deviceIdList, uint32_t deviceNums,
    aclprofAicoreMetrics aicoreMetrics, const aclprofAicoreEvents *aicoreEvents, uint64_t dataTypeConfig);
MSVP_PROF_API aclError aclprofDestroyConfig(const aclprofConfig *profilerConfig);
MSVP_PROF_API aclError aclprofStop(const aclprofConfig *profilerConfig);

#ifdef __cplusplus
}
#endif

#endif  // INC_EXTERNAL_ACL_PROF_H_
