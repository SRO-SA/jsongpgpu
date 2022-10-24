//compile with: g++ -std=c++17 -Wall -mavx2 -mpclmul -msse2 -mbmi  Basic_Tokenizer.cpp json_character_block.cpp -o a.exe
// simd.prev = in[248:128]prev[255:248]in[119:0]in[127:120];
// for backslashes overflow, the last two digits (most significant) of the bit array should be either 10 or 11, if its 10 there
// is definitely an overflow. Otherwise, find ~(a|b) (on the remaining bits) to get all 0 bits and a&b to get all 1 bits. If (a&b)>~(a|b)
// then we have an overflow. on the other hand, if they are equal then this means that there is only one set bit on both a and b, so we have to find
// the overflow of previous chunk.
// use prefix-xor to find the propagation negation on most significant bits.

#include <immintrin.h>

#include <emmintrin.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

//#include "simd.h"
//#include "dom_parser_implementation_real.h"
#include "json_character_block.h"

#ifndef simdjson_likely
#define simdjson_likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef simdjson_unlikely
#define simdjson_unlikely(x) __builtin_expect(!!(x), 0)
#endif

#include <cstdint>

SIMDJSON_DLLIMPORTEXPORT  const unsigned char BitsSetTable256mul2[256] = {
    0,  2,  2,  4,  2,  4,  4,  6,  2,  4,  4,  6,  4,  6,  6,  8,  2,  4,  4,
    6,  4,  6,  6,  8,  4,  6,  6,  8,  6,  8,  8,  10, 2,  4,  4,  6,  4,  6,
    6,  8,  4,  6,  6,  8,  6,  8,  8,  10, 4,  6,  6,  8,  6,  8,  8,  10, 6,
    8,  8,  10, 8,  10, 10, 12, 2,  4,  4,  6,  4,  6,  6,  8,  4,  6,  6,  8,
    6,  8,  8,  10, 4,  6,  6,  8,  6,  8,  8,  10, 6,  8,  8,  10, 8,  10, 10,
    12, 4,  6,  6,  8,  6,  8,  8,  10, 6,  8,  8,  10, 8,  10, 10, 12, 6,  8,
    8,  10, 8,  10, 10, 12, 8,  10, 10, 12, 10, 12, 12, 14, 2,  4,  4,  6,  4,
    6,  6,  8,  4,  6,  6,  8,  6,  8,  8,  10, 4,  6,  6,  8,  6,  8,  8,  10,
    6,  8,  8,  10, 8,  10, 10, 12, 4,  6,  6,  8,  6,  8,  8,  10, 6,  8,  8,
    10, 8,  10, 10, 12, 6,  8,  8,  10, 8,  10, 10, 12, 8,  10, 10, 12, 10, 12,
    12, 14, 4,  6,  6,  8,  6,  8,  8,  10, 6,  8,  8,  10, 8,  10, 10, 12, 6,
    8,  8,  10, 8,  10, 10, 12, 8,  10, 10, 12, 10, 12, 12, 14, 6,  8,  8,  10,
    8,  10, 10, 12, 8,  10, 10, 12, 10, 12, 12, 14, 8,  10, 10, 12, 10, 12, 12,
    14, 10, 12, 12, 14, 12, 14, 14, 16};

SIMDJSON_DLLIMPORTEXPORT  const uint8_t pshufb_combine_table[272] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x08,
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0xff,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
    0x0f, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0xff, 0xff, 0xff, 0x00, 0x01, 0x02, 0x08,
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0x01, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
    0x0f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
};

// 256 * 8 bytes = 2kB, easily fits in cache.
SIMDJSON_DLLIMPORTEXPORT  const uint64_t thintable_epi8[256] = {
    0x0706050403020100, 0x0007060504030201, 0x0007060504030200,
    0x0000070605040302, 0x0007060504030100, 0x0000070605040301,
    0x0000070605040300, 0x0000000706050403, 0x0007060504020100,
    0x0000070605040201, 0x0000070605040200, 0x0000000706050402,
    0x0000070605040100, 0x0000000706050401, 0x0000000706050400,
    0x0000000007060504, 0x0007060503020100, 0x0000070605030201,
    0x0000070605030200, 0x0000000706050302, 0x0000070605030100,
    0x0000000706050301, 0x0000000706050300, 0x0000000007060503,
    0x0000070605020100, 0x0000000706050201, 0x0000000706050200,
    0x0000000007060502, 0x0000000706050100, 0x0000000007060501,
    0x0000000007060500, 0x0000000000070605, 0x0007060403020100,
    0x0000070604030201, 0x0000070604030200, 0x0000000706040302,
    0x0000070604030100, 0x0000000706040301, 0x0000000706040300,
    0x0000000007060403, 0x0000070604020100, 0x0000000706040201,
    0x0000000706040200, 0x0000000007060402, 0x0000000706040100,
    0x0000000007060401, 0x0000000007060400, 0x0000000000070604,
    0x0000070603020100, 0x0000000706030201, 0x0000000706030200,
    0x0000000007060302, 0x0000000706030100, 0x0000000007060301,
    0x0000000007060300, 0x0000000000070603, 0x0000000706020100,
    0x0000000007060201, 0x0000000007060200, 0x0000000000070602,
    0x0000000007060100, 0x0000000000070601, 0x0000000000070600,
    0x0000000000000706, 0x0007050403020100, 0x0000070504030201,
    0x0000070504030200, 0x0000000705040302, 0x0000070504030100,
    0x0000000705040301, 0x0000000705040300, 0x0000000007050403,
    0x0000070504020100, 0x0000000705040201, 0x0000000705040200,
    0x0000000007050402, 0x0000000705040100, 0x0000000007050401,
    0x0000000007050400, 0x0000000000070504, 0x0000070503020100,
    0x0000000705030201, 0x0000000705030200, 0x0000000007050302,
    0x0000000705030100, 0x0000000007050301, 0x0000000007050300,
    0x0000000000070503, 0x0000000705020100, 0x0000000007050201,
    0x0000000007050200, 0x0000000000070502, 0x0000000007050100,
    0x0000000000070501, 0x0000000000070500, 0x0000000000000705,
    0x0000070403020100, 0x0000000704030201, 0x0000000704030200,
    0x0000000007040302, 0x0000000704030100, 0x0000000007040301,
    0x0000000007040300, 0x0000000000070403, 0x0000000704020100,
    0x0000000007040201, 0x0000000007040200, 0x0000000000070402,
    0x0000000007040100, 0x0000000000070401, 0x0000000000070400,
    0x0000000000000704, 0x0000000703020100, 0x0000000007030201,
    0x0000000007030200, 0x0000000000070302, 0x0000000007030100,
    0x0000000000070301, 0x0000000000070300, 0x0000000000000703,
    0x0000000007020100, 0x0000000000070201, 0x0000000000070200,
    0x0000000000000702, 0x0000000000070100, 0x0000000000000701,
    0x0000000000000700, 0x0000000000000007, 0x0006050403020100,
    0x0000060504030201, 0x0000060504030200, 0x0000000605040302,
    0x0000060504030100, 0x0000000605040301, 0x0000000605040300,
    0x0000000006050403, 0x0000060504020100, 0x0000000605040201,
    0x0000000605040200, 0x0000000006050402, 0x0000000605040100,
    0x0000000006050401, 0x0000000006050400, 0x0000000000060504,
    0x0000060503020100, 0x0000000605030201, 0x0000000605030200,
    0x0000000006050302, 0x0000000605030100, 0x0000000006050301,
    0x0000000006050300, 0x0000000000060503, 0x0000000605020100,
    0x0000000006050201, 0x0000000006050200, 0x0000000000060502,
    0x0000000006050100, 0x0000000000060501, 0x0000000000060500,
    0x0000000000000605, 0x0000060403020100, 0x0000000604030201,
    0x0000000604030200, 0x0000000006040302, 0x0000000604030100,
    0x0000000006040301, 0x0000000006040300, 0x0000000000060403,
    0x0000000604020100, 0x0000000006040201, 0x0000000006040200,
    0x0000000000060402, 0x0000000006040100, 0x0000000000060401,
    0x0000000000060400, 0x0000000000000604, 0x0000000603020100,
    0x0000000006030201, 0x0000000006030200, 0x0000000000060302,
    0x0000000006030100, 0x0000000000060301, 0x0000000000060300,
    0x0000000000000603, 0x0000000006020100, 0x0000000000060201,
    0x0000000000060200, 0x0000000000000602, 0x0000000000060100,
    0x0000000000000601, 0x0000000000000600, 0x0000000000000006,
    0x0000050403020100, 0x0000000504030201, 0x0000000504030200,
    0x0000000005040302, 0x0000000504030100, 0x0000000005040301,
    0x0000000005040300, 0x0000000000050403, 0x0000000504020100,
    0x0000000005040201, 0x0000000005040200, 0x0000000000050402,
    0x0000000005040100, 0x0000000000050401, 0x0000000000050400,
    0x0000000000000504, 0x0000000503020100, 0x0000000005030201,
    0x0000000005030200, 0x0000000000050302, 0x0000000005030100,
    0x0000000000050301, 0x0000000000050300, 0x0000000000000503,
    0x0000000005020100, 0x0000000000050201, 0x0000000000050200,
    0x0000000000000502, 0x0000000000050100, 0x0000000000000501,
    0x0000000000000500, 0x0000000000000005, 0x0000000403020100,
    0x0000000004030201, 0x0000000004030200, 0x0000000000040302,
    0x0000000004030100, 0x0000000000040301, 0x0000000000040300,
    0x0000000000000403, 0x0000000004020100, 0x0000000000040201,
    0x0000000000040200, 0x0000000000000402, 0x0000000000040100,
    0x0000000000000401, 0x0000000000000400, 0x0000000000000004,
    0x0000000003020100, 0x0000000000030201, 0x0000000000030200,
    0x0000000000000302, 0x0000000000030100, 0x0000000000000301,
    0x0000000000000300, 0x0000000000000003, 0x0000000000020100,
    0x0000000000000201, 0x0000000000000200, 0x0000000000000002,
    0x0000000000000100, 0x0000000000000001, 0x0000000000000000,
    0x0000000000000000,
}; //static uint64_t thintable_epi8[256]



#pragma once
#include <string>
#include <fstream>
#include <stdexcept>
 
namespace Utils {
    /**
    * FUNCTION: readAllText 
    * USE: Opens a text file, reads all the text in the file into a string, 
    *   and then closes the file
    * @param path: The file to open for reading
    * @return: A string containing all text in the file
    */
    std::string readAllText(const std::string& path) {
        std::ifstream infile;
        infile.open(path);
        if (infile.fail()) {
            throw std::invalid_argument{ "Unable to read file at path: '" + path + "'" };
        }
        infile.seekg(0, std::ios::end);
        auto size = infile.tellg();
        std::string contents((std::size_t)size, ' ');
        infile.seekg(0);
        infile.read(&contents[0], size);
        return contents;
    }
 
    /**
    * FUNCTION: writeAllText 
    * USE: Creates a new file, writes the specified string to the file, 
    *   and then closes the file. If the target file already exists, it 
    *   is overwritten
    * @param path: The file to write to
    * @param contents: The string to write to the file
    * @return: N/A
    */
    void writeAllText(const std::string& path, const std::string& contents) {
        std::ofstream outfile;
        outfile.open(path);
        if (outfile.fail()) {
            throw std::invalid_argument{ "Unable to write file at path: '" + path + "'" };
        }
        outfile << contents;
    }
}// http://programmingnotes.org/



int main(){
    json_character_block b = json_character_block();
    dom_parser_implementation p;
    clock_t start, end;
    double time = 0;

    p.set_capacity(SIMDJSON_MAXSIZE_BYTES);
    p.set_max_depth(DEFAULT_MAX_DEPTH);
    // Declare path
    std::string path = "./../../Large-Json/google_map_small_records.json";
 
    // Read the contents of the file at the specified path
    std::string str = Utils::readAllText(path);

    //std::string str = "{\"k1\":\"v1\",\"k2\":\"v2\",\"k3\":\"v3\",\"k4\":{\"k5\":\"v4\",\"k6\":\"v5\"},\"k7\":[\"v6\",\"v7\",[]],\"k8\":1234}";
    start = clock();
    bool isValid = validate_utf8(str.c_str(), str.size());
    end = clock();
    time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //std::cout << isValid << std::endl;
    std::cout << "Validation Time elapsed: " << time << std::endl;
    start = clock();
    isValid = p.stage1((uint8_t *) str.c_str(), str.size(), false);
    end = clock();

    time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    std::cout << "Stage1 Time elapsed: " << time << std::endl;

    //std::cout << isValid << std::endl;
    //std::cout << str.size() << std::endl;
    double total= 0;
    for(uint32_t i=0; i< p.n_structural_indexes; i++){
        start = clock();
        uint32_t *v = p.structural_indexes.get();
        end = clock();
        total += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
        //printf("%08" PRIx32 " ", v[i]);
        //std::bitset<32> t(v[i]);
        //std::cout << t << std::endl;
        //std::cout << i << std::endl;
    }
    printf("%f\n", total);
    //simd8<uint8_t> buf = simd8<uint8_t>(str)
    return 0;
}

