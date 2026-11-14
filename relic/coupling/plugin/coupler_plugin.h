#ifndef RELIC_COUPLER_PLUGIN_H
#define RELIC_COUPLER_PLUGIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COUPLER_PLUGIN_VERSION_MAJOR 1
#define COUPLER_PLUGIN_VERSION_MINOR 0

typedef enum {
    CP_INT32 = 0,
    CP_INT64 = 1,
    CP_F32 = 2,
    CP_F64 = 3,
    CP_STR = 4,
    CP_BYTES = 5
} cp_type_id_t;

/* Probe plugin capabilities; writes NUL-terminated JSON manifest. */
int plugin_probe(char* manifest_json_out, size_t buf_len);

/* Negotiate plugin-host capabilities; writes NUL-terminated agreed manifest. */
int plugin_negotiate(
    const char* host_capabilities_json,
    char* agreed_manifest_out,
    size_t buf_len
);

/* Exchange data buffer for a named port at the provided timestamp. */
int plugin_exchange(
    void* handle,
    const char* port_name,
    void* buffer_ptr,
    size_t buffer_len,
    double timestamp_seconds
);

#ifdef __cplusplus
}
#endif

#endif /* RELIC_COUPLER_PLUGIN_H */
