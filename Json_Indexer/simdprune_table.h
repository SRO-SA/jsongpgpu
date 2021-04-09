#ifndef SIMDJSON_INTERNAL_SIMDPRUNE_TABLES_H
#define SIMDJSON_INTERNAL_SIMDPRUNE_TABLES_H

#include <cstdint>
#include "common_defs.h"

extern SIMDJSON_DLLIMPORTEXPORT const unsigned char BitsSetTable256mul2[256];

extern SIMDJSON_DLLIMPORTEXPORT const uint8_t pshufb_combine_table[272];

// 256 * 8 bytes = 2kB, easily fits in cache.
extern SIMDJSON_DLLIMPORTEXPORT const uint64_t thintable_epi8[256];


#endif // SIMDJSON_INTERNAL_SIMDPRUNE_TABLES_H