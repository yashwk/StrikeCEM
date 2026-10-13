#ifndef STRIKECEM_C_API_H
#define STRIKECEM_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef SCEM_BUILDING_DLL
    #define SCEM_API __declspec(dllexport)
  #else
    #define SCEM_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define SCEM_API __attribute__((visibility("default")))
  #else
    #define SCEM_API
  #endif
#endif

typedef enum {
    SCEM_OK = 0,
    SCEM_ERROR_INVALID_ARGUMENT = 1,
    SCEM_ERROR_FILE_NOT_FOUND = 2,
    SCEM_ERROR_CORRUPT_DATABASE = 3,
    SCEM_ERROR_INCOMPLETE_DATABASE = 4,
    SCEM_ERROR_DESIGN_MISMATCH = 5,
    SCEM_ERROR_CHANNEL_NOT_FOUND = 6,
    SCEM_ERROR_INTERNAL = 7
} scem_status_t;

typedef struct {
    double scattering_real;
    double scattering_imag;
    double rcs_sqm;
    double rcs_dbsm;
} scem_query_result_t;

typedef struct {
    double azimuth_deg;
    double elevation_deg;
    double frequency_hz;
    char polarization[8];
} scem_query_request_t;

typedef struct scem_database scem_database_t;

SCEM_API scem_status_t scem_database_open(const char* path, scem_database_t** out_db);

SCEM_API void scem_database_close(scem_database_t* db);

SCEM_API scem_status_t scem_database_get_design_id(scem_database_t* db, char* buf, size_t buf_size);

SCEM_API scem_status_t scem_database_get_design_revision(scem_database_t* db, char* buf, size_t buf_size);

SCEM_API scem_status_t scem_database_verify_design(scem_database_t* db, const char* expected_design_id, const char* expected_revision);

SCEM_API scem_status_t scem_database_get_frequencies(scem_database_t* db, double* out_freqs, size_t* inout_count);

SCEM_API scem_status_t scem_database_get_polarizations(scem_database_t* db, char out_pols[][8], size_t* inout_count);

SCEM_API scem_status_t scem_database_query(scem_database_t* db, double az_deg, double el_deg, double freq_hz, const char* pol, scem_query_result_t* out_result);

SCEM_API scem_status_t scem_database_query_batch(scem_database_t* db, const scem_query_request_t* requests, size_t count, scem_query_result_t* out_results);

SCEM_API const char* scem_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // STRIKECEM_C_API_H
