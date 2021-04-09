#ifndef SIMDJSON_DOM_DOCUMENT_H
#define SIMDJSON_DOM_DOCUMENT_H

#include "common_defs.h"
#include <memory>
#include <ostream>

enum error_code {
  SUCCESS = 0,              ///< No error
  CAPACITY,                 ///< This parser can't support a document that big
  MEMALLOC,                 ///< Error allocating memory, most likely out of memory
  TAPE_ERROR,               ///< Something went wrong while writing to the tape (stage 2), this is a generic error
  DEPTH_ERROR,              ///< Your document exceeds the user-specified depth limitation
  STRING_ERROR,             ///< Problem while parsing a string
  T_ATOM_ERROR,             ///< Problem while parsing an atom starting with the letter 't'
  F_ATOM_ERROR,             ///< Problem while parsing an atom starting with the letter 'f'
  N_ATOM_ERROR,             ///< Problem while parsing an atom starting with the letter 'n'
  NUMBER_ERROR,             ///< Problem while parsing a number
  UTF8_ERROR,               ///< the input is not valid UTF-8
  UNINITIALIZED,            ///< unknown error, or uninitialized document
  EMPTY,                    ///< no structural element found
  UNESCAPED_CHARS,          ///< found unescaped characters in a string.
  UNCLOSED_STRING,          ///< missing quote at the end
  UNSUPPORTED_ARCHITECTURE, ///< unsupported architecture
  INCORRECT_TYPE,           ///< JSON element has a different type than user expected
  NUMBER_OUT_OF_RANGE,      ///< JSON number does not fit in 64 bits
  INDEX_OUT_OF_BOUNDS,      ///< JSON array index too large
  NO_SUCH_FIELD,            ///< JSON field not found in object
  IO_ERROR,                 ///< Error reading a file
  INVALID_JSON_POINTER,     ///< Invalid JSON pointer reference
  INVALID_URI_FRAGMENT,     ///< Invalid URI fragment
  UNEXPECTED_ERROR,         ///< indicative of a bug in simdjson
  PARSER_IN_USE,            ///< parser is already in use.
  OUT_OF_ORDER_ITERATION,   ///< tried to iterate an array or object out of order
  INSUFFICIENT_PADDING,     ///< The JSON doesn't have enough padding for simdjson to safely parse it.
  NUM_ERROR_CODES
};

namespace dom {

class element;

/**
 * A parsed JSON document.
 *
 * This class cannot be copied, only moved, to avoid unintended allocations.
 */
class document {
public:
  /**
   * Create a document container with zero capacity.
   *
   * The parser will allocate capacity as needed.
   */
  document() noexcept = default;
  ~document() noexcept = default;

  /**
   * Take another document's buffers.
   *
   * @param other The document to take. Its capacity is zeroed and it is invalidated.
   */
  document(document &&other) noexcept = default;
  /** @private */
  document(const document &) = delete; // Disallow copying
  /**
   * Take another document's buffers.
   *
   * @param other The document to take. Its capacity is zeroed.
   */
  document &operator=(document &&other) noexcept = default;
  /** @private */
  document &operator=(const document &) = delete; // Disallow copying

  /**
   * Get the root element of this document as a JSON array.
   */
  element root() const noexcept;

  /**
   * @private Dump the raw tape for debugging.
   *
   * @param os the stream to output to.
   * @return false if the tape is likely wrong (e.g., you did not parse a valid JSON).
   */
  bool dump_raw_tape(std::ostream &os) const noexcept;

  /** @private Structural values. */
  std::unique_ptr<uint64_t[]> tape{};

  /** @private String values.
   *
   * Should be at least byte_capacity.
   */
  std::unique_ptr<uint8_t[]> string_buf{};
  /** @private Allocate memory to support
   * input JSON documents of up to len bytes.
   *
   * When calling this function, you lose
   * all the data.
   *
   * The memory allocation is strict: you
   * can you use this function to increase
   * or lower the amount of allocated memory.
   * Passsing zero clears the memory.
   */
  error_code allocate(size_t len) noexcept;
  /** @private Capacity in bytes, in terms
   * of how many bytes of input JSON we can
   * support.
   */
  size_t capacity() const noexcept;


private:
  size_t allocated_capacity{0};
  friend class parser;
}; // class document

} // namespace dom

#endif // SIMDJSON_DOM_DOCUMENT_H