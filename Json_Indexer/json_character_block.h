#ifndef JSON_CHARACTER_BLOCK_H
#define JSON_CHARACTER_BLOCK_H

#include "common_defs.h"
#include "simd.h"
#include "implementation.h"
#include "buf_block_reader.h"
#include <bitset>

simdjson_really_inline bool is_ascii(const simd8x64<uint8_t>& input) ;
simdjson_really_inline simd8<bool> must_be_2_3_continuation(const simd8<uint8_t> prev2, const simd8<uint8_t> prev3);
template<class checker>
bool generic_validate_utf8(const uint8_t * input, size_t length);
bool generic_validate_utf8(const char * input, size_t length);
struct json_character_block {
  static simdjson_really_inline json_character_block classify(const simd8x64<uint8_t>& in);
  //  ASCII white-space ('\r','\n','\t',' ')
  simdjson_really_inline uint64_t whitespace() const noexcept;
  // non-quote structural characters (comma, colon, braces, brackets)
  simdjson_really_inline uint64_t op() const noexcept;
  // neither a structural character nor a white-space, so letters, numbers and quotes
  simdjson_really_inline uint64_t scalar() const noexcept;

  uint64_t _whitespace; // ASCII white-space ('\r','\n','\t',' ')
  uint64_t _op; // structural characters (comma, colon, braces, brackets but not quotes)
};
simdjson_unused simdjson_really_inline simd8<bool> must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2, const simd8<uint8_t> prev3);
simdjson_warn_unused bool validate_utf8(const char *buf, size_t len) noexcept;

simdjson_really_inline bool is_ascii(const simd8x64<uint8_t>& input) {
  return input.reduce_or().is_ascii();
}



simdjson_really_inline uint64_t json_character_block::whitespace() const noexcept { return _whitespace; }
simdjson_really_inline uint64_t json_character_block::op() const noexcept { return _op; }
simdjson_really_inline uint64_t json_character_block::scalar() const noexcept { return ~(op() | whitespace()); }


// This identifies structural characters (comma, colon, braces, brackets),
// and ASCII white-space ('\r','\n','\t',' ').
simdjson_really_inline json_character_block json_character_block::classify(const simd8x64<uint8_t>& in) {
  // These lookups rely on the fact that anything < 127 will match the lower 4 bits, which is why
  // we can't use the generic lookup_16.
  const auto whitespace_table = simd8<uint8_t>::repeat_16(' ', 100, 100, 100, 17, 100, 113, 2, 100, '\t', '\n', 112, 100, '\r', 100, 100);

  // The 6 operators (:,[]{}) have these values:
  //
  // , 2C
  // : 3A
  // [ 5B
  // { 7B
  // ] 5D
  // } 7D
  //
  // If you use | 0x20 to turn [ and ] into { and }, the lower 4 bits of each character is unique.
  // We exploit this, using a simd 4-bit lookup to tell us which character match against, and then
  // match it (against | 0x20).
  //
  // To prevent recognizing other characters, everything else gets compared with 0, which cannot
  // match due to the | 0x20.
  //
  // NOTE: Due to the | 0x20, this ALSO treats <FF> and <SUB> (control characters 0C and 1A) like ,
  // and :. This gets caught in stage 2, which checks the actual character to ensure the right
  // operators are in the right places.
  const auto op_table = simd8<uint8_t>::repeat_16(
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, ':', '{', // : = 3A, [ = 5B, { = 7B
    ',', '}', 0, 0 ) // , = 2C, ] = 5D, } = 7D repeat_16ne or the
  // other, given the fact that all functions are aggressively inlined, we can
  // hope that useless computations will be omitted. This is namely case when
  // minifying (we only need whitespace).

  const uint64_t whitespace = in.eq({
    _mm256_shuffle_epi8(whitespace_table, in.chunks[0]),
    _mm256_shuffle_epi8(whitespace_table, in.chunks[1])
  });
  // Turn [ and ] into { and }
  const simd8x64<uint8_t> curlified{
    in.chunks[0] | 0x20,
    in.chunks[1] | 0x20
  };
  const uint64_t op = curlified.eq({
    _mm256_shuffle_epi8(op_table, in.chunks[0]),
    _mm256_shuffle_epi8(op_table, in.chunks[1])
  });
  return { whitespace, op };
}




simdjson_really_inline simd8<bool> must_be_2_3_continuation(const simd8<uint8_t> prev2, const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_third_byte  = prev2.saturating_sub(0b11100000u-1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte = prev3.saturating_sub(0b11110000u-1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_third_byte | is_fourth_byte) > int8_t(0);
}
#include "structural_indexer.h"
#include "json_string_scanner.h"

#include "utf8_lookup4_algorithm.h"

#endif