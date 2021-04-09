


#include "json_character_block.h"



template<class checker>
bool generic_validate_utf8(const uint8_t * input, size_t length) {
    utf8_checker c{};
    buf_block_reader<64> reader(input, length);
    while (reader.has_full_block()) {
      simd8x64<uint8_t> in(reader.full_block());
      c.check_next_input(in);
      reader.advance();
    }
    uint8_t block[64]{};
    reader.get_remainder(block);
    simd8x64<uint8_t> in(block);
    c.check_next_input(in);
    reader.advance();
    c.check_eof();
    //std::cout << "Errors: " << c.errors() << std::endl;
    return c.errors() == error_code::SUCCESS;
}

bool generic_validate_utf8(const char * input, size_t length) {
    return generic_validate_utf8<utf8_checker>(reinterpret_cast<const uint8_t *>(input),length);
}

//
// Stage 1
//
#include "structural_indexer.h"


simdjson_really_inline uint64_t json_string_scanner::find_escaped(uint64_t backslash) {
  if (!backslash) { uint64_t escaped = prev_escaped; prev_escaped = 0; return escaped; }
  return find_escaped_branchless(backslash);
}



simdjson_warn_unused error_code implementation::minify(const uint8_t *buf, size_t len, uint8_t *dst, size_t &dst_len) const noexcept {
  return json_minifier::minify<128>(buf, len, dst, dst_len);
}

simdjson_warn_unused error_code dom_parser_implementation::stage1(const uint8_t *_buf, size_t _len, bool streaming) noexcept {
  this->buf = _buf;
  this->len = _len;
  return json_structural_indexer::index<128>(_buf, _len, *this, streaming);
}

simdjson_warn_unused bool implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return generic_validate_utf8(buf,len);
}

// simdjson_warn_unused error_code dom_parser_implementation::stage2(dom::document &_doc) noexcept {
//   return stage2::tape_builder::parse_document<false>(*this, _doc);
// }

// simdjson_warn_unused error_code dom_parser_implementation::stage2_next(dom::document &_doc) noexcept {
//   return stage2::tape_builder::parse_document<true>(*this, _doc);
// }

simdjson_warn_unused error_code dom_parser_implementation::parse(const uint8_t *_buf, size_t _len) noexcept {
  auto error = stage1(_buf, _len, false);
  if (error) { return error; }
  return error;
  //return stage2(_doc);
}

simdjson_warn_unused error_code implementation::create_dom_parser_implementation(
  size_t capacity,
  size_t max_depth,
  std::unique_ptr<dom_parser_implementation>& dst
) const noexcept {
  dst.reset( new (std::nothrow) dom_parser_implementation() );
  if (!dst) { return MEMALLOC; }
  dst->set_capacity(capacity);
  dst->set_max_depth(max_depth);
  return SUCCESS;
}


simdjson_unused simdjson_really_inline simd8<bool> must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2, const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_second_byte = prev1.saturating_sub(0b11000000u-1); // Only 11______ will be > 0
  simd8<uint8_t> is_third_byte  = prev2.saturating_sub(0b11100000u-1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte = prev3.saturating_sub(0b11110000u-1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_second_byte | is_third_byte | is_fourth_byte) > int8_t(0);
}


simdjson_warn_unused bool validate_utf8(const char *buf, size_t len) noexcept {
  return generic_validate_utf8(buf,len);
}

