#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <chrono>
#include <thread>
#include <x86intrin.h>
#include "cuda_profiler_api.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <string.h>
#include <pthread.h>
#include <bitset>
#include "JSON_Parallel_Parser_Threads.h"
#include <thrust/transform.h>
#include <inttypes.h>

#define        MAXLINELENGTH    268435456   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824// Max record size
                                            //4MB       8MB     32BM    64MB      128MB    192MB     256MB     512MB     768MB       1GB
#define        BUFSIZE          268435456   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824

#define AVGGPUCLOCK 1346000000


enum hypothesis_val {in, out, unknown, fail};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define RUNTIMES 1

#define BLOCKSIZE 128
#define FILESCOUNT 4
#define NAMELENGTH 25

#define CPUTHREADS 8

#define OPENBRACKET 91
#define CLOSEBRACKET 93
#define OPENBRACE 123
#define CLOSEBRACE 125
#define COMMA 44
#define I 73

#define ROW1 1
#define ROW2 2
#define ROW3 3
#define ROW4 4
#define ROW5 5
#define ROW6 6

struct start_input_t
{
    //start_input_t(double total): total_runtime(total) {};
    //start_input_t(){total_runtime = 0;};
    uint64_t size;
    static std::atomic<double> total_runtime;
    uint8_t * block;
    uint32_t* res;
};

std::atomic<double> start_input_t::total_runtime(0);


struct not_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x > 0);
    }
};

struct is_newline
{
    __host__ __device__
    bool operator()(const char x)
    {
        return (x == '\n');
    }
};

struct is_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x == 0);
    }
};


struct s_not_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x != 0);
    }
};

struct not_ff
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x < 0xffffffff);
    }
};

struct not_minus
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x > -1);
    }
};

typedef uint32_t(*fptr_t)(const uint32_t&);

template <fptr_t F>
struct functor{

  __host__ __device__ uint32_t operator()(const uint32_t& x) const {
        return F(x);
    }
};

__host__ __device__ uint32_t set_to_zero(const uint32_t& x) { return 0;}



int print_array(long* input, int length, int rows){
    for(long i =0; i<rows; i++){
      for(long j=0; j<length; j++){
        std::cout << *(input+j+(i*length)) << ' ';
      }
      std::cout << std::endl;
    }
    return 1;
  }

// int print(uint32_t* input, int length, int rows){
//     for(long i =0; i<rows; i++){
//       for(long j=0; j<length; j++){
//         std::cout << *(input+j+(i*length)) << ' ';
//       }
//       std::cout << std::endl;
//     }
//     return 1;
// }

int print_d(uint32_t* input_d, int length, int rows){
    uint32_t * input;
    input = (uint32_t*) malloc(sizeof(uint32_t)*length*rows);
    cudaMemcpyAsync(input, input_d, sizeof(uint32_t)*length*rows, cudaMemcpyDeviceToHost);

    for(long i =0; i<rows; i++){
      for(long j=0; j<length; j++){
        std::bitset<32> y(*(input+j+(i*length)));
        if(j == 129) printf("----129----");
        std::cout << y << ' ';
      }
      std::cout << std::endl;
    }
    return 1;
}

int print8(uint8_t* input, int length, int rows){
    for(long i =0; i<rows; i++){
        for(long j=0; j<length; j++){
            std::cout << *(input+j+(i*length)) << ' ';
        }
        std::cout << std::endl;
    }
    return 1;
}

template<typename T>
int print8_d(uint8_t* input_d, int length, int rows){

    uint8_t * input;
    input = (uint8_t*) malloc(sizeof(uint8_t)*length);
    cudaMemcpyAsync(input, input_d, sizeof(uint8_t)*length, cudaMemcpyDeviceToHost);

    for(long i =0; i<rows; i++){
        for(long j=0; j<length; j++){
            std::cout << (T )*(input+j+(i*length)) << ' ';
        }
        std::cout << std::endl;
    }
    return 1;
}

__global__
void parallel_shift_left(uint32_t* input, uint32_t* output, uint32_t bits, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        output[i] = (input[i] << bits); 
    }
}

__global__
void parallel_shift_right(uint32_t* input, uint32_t* output, uint32_t bits, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        output[i] = (input[i] >> bits); 
    }
}

__global__
void parallel_not(uint32_t* input, uint32_t* output, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        output[i] = ~input[i]; 
    }
}

template<typename T>
__global__
void single_parallel_and(T* input, uint8_t value, T* output, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        output[i] = input[i] & ((uint32_t)value | (uint32_t)value << 8 | (uint32_t)value << 16 | (uint32_t)value << 24);
    }
}

template<typename T>
__global__ 
void parallel_and(T *input1, T *input2, T* output, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        output[i] = input1[i] & input2[i]; 
    }
}

__global__ 
void parallel_or(uint32_t *input1, uint32_t *input2, uint32_t* output, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
            output[i] = input1[i] | input2[i]; 
    }
}

__global__ 
void parallel_xor(uint32_t *input1, uint32_t *input2, uint32_t* output, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
            /*if(i==0){
                printf("input1: %d\n", input1[i]);
                printf("input2: %d\n", input2[i]);
            }*/
            output[i] = input1[i] ^ input2[i];
            //if(output[i]!=0) printf("index: %d, input1: %d, input2 %d\n", i, input1[i], input2[i]); 
    }
}

__device__ __forceinline__
void must_be_2_3_continuation_parallel_and_parallel_xor(uint32_t prev1, uint32_t prev2, uint32_t sc, uint32_t& must32_80_sc, uint64_t size, int total_padded_32){
    static const uint32_t third_subtract_byte = (0b11100000u-1) | (0b11100000u-1) << 8 | (0b11100000u-1) << 16 | (0b11100000u-1) << 24;
    static const uint32_t fourth_subtract_byte = (0b11110000u-1) | (0b11110000u-1) << 8 | (0b11110000u-1) << 16 | (0b11110000u-1) << 24;
    // int prev = (i-1)*4;
    // int start = i*4;

    uint32_t is_third_byte = __vsubus4(prev1, third_subtract_byte);
    uint32_t is_fourth_byte = __vsubus4(prev2, fourth_subtract_byte);
    /*if(i==8880){
        printf("Third %d\n", is_third_byte);
        printf("Fourth %d\n", is_fourth_byte);
    }*/

    uint32_t gt = __vsubss4((int32_t)(is_third_byte | is_fourth_byte), int32_t(0));
    gt = gt & 0xFFFFFFFF;
    uint32_t must32 = __vcmpgtu4(gt, 0);
    uint32_t must32_80 = must32 & 0x80808080;
    must32_80_sc = must32_80 ^ sc;

    // uint8_t gt1 = (int8_t)gt & 0xFF > 0 ? 0xFF : 0;
    // uint8_t gt2 =  (int8_t)(gt >> 8) & 0xFF > 0 ? 0xFF : 0;
    // uint8_t gt3 =  (int8_t)(gt >> 16) & 0xFF > 0 ? 0xFF : 0;
    // uint8_t gt4 =  (int8_t)(gt >> 24) & 0xFF > 0 ? 0xFF : 0;

    // must32_d[i] = gt1 | (gt2 << 8) | (gt3 << 16) | (gt4 << 24);
    /*if(i==0){
        printf("gt: %u\n", gt);
    }*/
}

__device__ __forceinline__
void check_special_cases(uint32_t block_compressed, uint32_t prev1, uint32_t& sc, uint64_t size, int total_padded_32){
    constexpr const uint8_t TOO_SHORT   = 1<<0; // 11______ 0_______
                                                // 11______ 11______
    constexpr const uint8_t TOO_LONG    = 1<<1; // 0_______ 10______
    constexpr const uint8_t OVERLONG_3  = 1<<2; // 11100000 100_____
    constexpr const uint8_t SURROGATE   = 1<<4; // 11101101 101_____
    constexpr const uint8_t OVERLONG_2  = 1<<5; // 1100000_ 10______
    constexpr const uint8_t TWO_CONTS   = 1<<7; // 10______ 10______
    constexpr const uint8_t TOO_LARGE   = 1<<3; // 11110100 1001____
                                                // 11110100 101_____
                                                // 11110101 1001____
                                                // 11110101 101_____
                                                // 1111011_ 1001____
                                                // 1111011_ 101_____
                                                // 11111___ 1001____
                                                // 11111___ 101_____
    constexpr const uint8_t TOO_LARGE_1000 = 1<<6;
                                                // 11110101 1000____
                                                // 1111011_ 1000____
                                                // 11111___ 1000____
    constexpr const uint8_t OVERLONG_4  = 1<<6; // 11110000 1000____
    constexpr const uint8_t CARRY = TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 . 10000011

    constexpr const uint32_t TOO_SHORT_32 = (((uint32_t)TOO_SHORT) | ((uint32_t)TOO_SHORT) << 8 | ((uint32_t)TOO_SHORT) << 16 | ((uint32_t)TOO_SHORT) << 24);
    constexpr const uint32_t TOO_LONG_32 = (((uint32_t)TOO_LONG) | ((uint32_t)TOO_LONG) << 8 | ((uint32_t)TOO_LONG) << 16 | ((uint32_t)TOO_LONG) << 24);
    constexpr const uint32_t OVERLONG_3_32 = (((uint32_t)OVERLONG_3) | ((uint32_t)OVERLONG_3) << 8 | ((uint32_t)OVERLONG_3) << 16 | ((uint32_t)OVERLONG_3) << 24);
    constexpr const uint32_t SURROGATE_32 = (((uint32_t)SURROGATE) | ((uint32_t)SURROGATE) << 8 | ((uint32_t)SURROGATE) << 16 | ((uint32_t)SURROGATE) << 24);
    constexpr const uint32_t OVERLONG_2_32 = (((uint32_t)OVERLONG_2) | ((uint32_t)OVERLONG_2) << 8 | ((uint32_t)OVERLONG_2) << 16 | ((uint32_t)OVERLONG_2) << 24);
    constexpr const uint32_t TWO_CONTS_32 = (((uint32_t)TWO_CONTS) | ((uint32_t)TWO_CONTS) << 8 | ((uint32_t)TWO_CONTS) << 16 | ((uint32_t)TWO_CONTS) << 24);
    constexpr const uint32_t TOO_LARGE_32 = (((uint32_t)TOO_LARGE) | ((uint32_t)TOO_LARGE) << 8 | ((uint32_t)TOO_LARGE) << 16 | ((uint32_t)TOO_LARGE) << 24);
    constexpr const uint32_t TOO_LARGE_1000_32 = (((uint32_t)TOO_LARGE_1000) | ((uint32_t)TOO_LARGE_1000) << 8 | ((uint32_t)TOO_LARGE_1000) << 16 | ((uint32_t)TOO_LARGE_1000) << 24);
    constexpr const uint32_t OVERLONG_4_32 = (((uint32_t)OVERLONG_4) | ((uint32_t)OVERLONG_4) << 8 | ((uint32_t)OVERLONG_4) << 16 | ((uint32_t)OVERLONG_4) << 24);
    constexpr const uint32_t CARRY_32 = (((uint32_t)CARRY) | ((uint32_t)CARRY) << 8 | ((uint32_t)CARRY) << 16 | ((uint32_t)CARRY) << 24);


          

    // uint8_t table1_g[16] = {      
    //     // 0_______ ________ <ASCII in byte 1>
    //     TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
    //     TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
    //     // 10______ ________ <continuation in byte 1>
    //     TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
    //     // 1100____ ________ <two byte lead in byte 1>
    //     TOO_SHORT | OVERLONG_2,
    //     // 1101____ ________ <two byte lead in byte 1>
    //     TOO_SHORT,
    //     // 1110____ ________ <three byte lead in byte 1>
    //     TOO_SHORT | OVERLONG_3 | SURROGATE,
    //     // 1111____ ________ <four+ byte lead in byte 1>
    //     TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4
    // };
    // uint8_t table2_g[16] = {
    //       // ____0000 ________
    //       CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4, //11100111
    //       // ____0001 ________
    //       CARRY | OVERLONG_2,
    //       // ____001_ ________
    //       CARRY,
    //       CARRY,
    
    //       // ____0100 ________
    //       CARRY | TOO_LARGE,
    //       // ____0101 ________
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       // ____011_ ________
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    
    //       // ____1___ ________
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       // ____1101 ________
    //       CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
    //       CARRY | TOO_LARGE | TOO_LARGE_1000,
    //       CARRY | TOO_LARGE | TOO_LARGE_1000
    // };
    // uint8_t table3_g[16] = {
    //   // ________ 0_______ <ASCII in byte 2>
    //   TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
    //   TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,

    //   // ________ 1000____
    //   TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
    //   // ________ 1001____
    //   TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
    //   // ________ 101_____
    //   TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE  | TOO_LARGE,
    //   TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE  | TOO_LARGE,

    //   // ________ 11______
    //   TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT
    // };
    // __shared__ uint8_t table1[16];
    // __shared__ uint8_t table2[16];
    // __shared__ uint8_t table3[16];
    // //tid < 16 ? table1[tid] = table1_g[tid] : NULL;
    // //tid < 16 ? table2[tid] = table2_g[tid] : NULL;
    // //tid < 16 ? table3[tid] = table3_g[tid] : NULL;
    // if(tid == 0){
    //     for(int k=0; k<16; k++){
    //         table1[k] = table1_g[k];
    //         table2[k] = table2_g[k];
    //         table3[k] = table3_g[k];
    //     }
    // }
    // __syncthreads();
    //int prev = (i-1)*4;
    // int start = i*4;
    uint32_t shr_prev1 = (prev1 >> 4) & 0x0f0f0f0f;
    //if(i==8880) printf("prev1_d: %u\n", prev1_d[i]);

    //if(i==8880) printf("shr_prev1: %u\n", shr_prev1);

    // uint8_t shr_prev1_b1 = shr_prev1 & 0x000000ff;
    // uint8_t shr_prev1_b2 = (shr_prev1 >> 8) & 0x000000ff;
    // uint8_t shr_prev1_b3 = (shr_prev1 >> 16) & 0x000000ff;
    // uint8_t shr_prev1_b4 = (shr_prev1 >> 24) & 0x000000ff;

    // uint32_t test1 = table1[shr_prev1_b1] | (table1[shr_prev1_b2] << 8) | (table1[shr_prev1_b3] << 16) | (table1[shr_prev1_b4] << 24);

    uint32_t shl_prev1 = prev1 & 0x0f0f0f0f;
    //  shr_prev1_b1 = shl_prev1;
    //  shr_prev1_b2 = (shl_prev1 >> 8);
    //  shr_prev1_b3 = (shl_prev1 >> 16);
    //  shr_prev1_b4 = (shl_prev1 >> 24);

    // uint32_t test2 = table2[shr_prev1_b1] | (table2[shr_prev1_b2] << 8) | (table2[shr_prev1_b3] << 16) | (table2[shr_prev1_b4] << 24);
    //if(i==8880) printf("shl_prev1: %u\n", shl_prev1);

    // shr_prev1_b1 = start<size ? (block_d[start] >> 4) : 0;
    // shr_prev1_b2 = start+1<size ? (block_d[start+1] >> 4) : 0;
    // shr_prev1_b3 = start+2<size ? (block_d[start+2] >> 4) : 0;
    // shr_prev1_b4 = start+3<size ? (block_d[start+3] >> 4) : 0;

    // uint32_t test3 = table3[shr_prev1_b1] | (table3[shr_prev1_b2] << 8) | (table3[shr_prev1_b3] << 16) | (table3[shr_prev1_b4] << 24);


    uint32_t byte_1_high = (__vcmpltu4(shr_prev1, 0x08080808) & TOO_LONG_32) | (__vcmpgeu4(shr_prev1, 0x08080808) & __vcmpltu4(shr_prev1, 0x0C0C0C0C) & TWO_CONTS_32) | (__vcmpgeu4(shr_prev1, 0x0C0C0C0C) & TOO_SHORT_32) 
    | (__vcmpeq4(shr_prev1, 0x0C0C0C0C) & OVERLONG_2_32) | (__vcmpeq4(shr_prev1, 0x0E0E0E0E) & (OVERLONG_3_32 | SURROGATE_32)) | (__vcmpeq4(shr_prev1, 0x0F0F0F0F) & (TOO_LARGE_32 | TOO_LARGE_1000_32 | OVERLONG_4_32));

    uint32_t byte_1_low = CARRY_32 | (__vcmpltu4(shl_prev1, 0x02020202) & OVERLONG_2_32) | (__vcmpgeu4(shl_prev1, 0x04040404) & TOO_LARGE_32) | (__vcmpgtu4(shl_prev1, 0x04040404) & TOO_LARGE_1000_32) 
            | (__vcmpeq4(shl_prev1, 0) & (OVERLONG_3_32 | OVERLONG_4_32)) | (__vcmpeq4(shl_prev1, 0x0D0D0D0D) & SURROGATE_32);
    
    uint32_t block_compressed_high = (block_compressed >> 4) & 0x0F0F0F0F;
    uint32_t less_than_12 = __vcmpltu4(block_compressed_high, 0x0C0C0C0C);

    uint32_t byte_2_high = ((__vcmpltu4(block_compressed_high, 0x08080808) | __vcmpgtu4(block_compressed_high, 0x0B0B0B0B)) & TOO_SHORT_32) | (less_than_12 & __vcmpgeu4(block_compressed_high, 0x08080808) & (TOO_LONG_32 | OVERLONG_2_32 | TWO_CONTS_32)) 
            | (less_than_12 & __vcmpgtu4(block_compressed_high, 0x08080808) & TOO_LARGE_32) | (__vcmpeq4(block_compressed_high, 0x08080808) & (TOO_LARGE_1000_32 | OVERLONG_4_32)) 
            | (__vcmpgtu4(block_compressed_high, 0x09090909) & less_than_12 & SURROGATE_32); 


    // if(i == 8880 && (test1 != byte_1_high || test2 != byte_1_low || test3 != byte_2_high)) printf("THERE IS AN ERROR IN check_special_cases !!!\n %x, %x, %x,\n %x, %x, %x,\n %x, %x, %x\n", shr_prev1, shl_prev1, block_compressed_high, test1, test2, test3, byte_1_high, byte_1_low, byte_2_high);
    sc =   (byte_1_high & byte_1_low & byte_2_high);
    //if(i==8880) printf("8880: %d, byte_1_high: %u, byte_1_low: %u, byte_2_high: %u\n", sc_d[i], byte_1_high, byte_1_low, byte_2_high);
    // if(i==8880) printf("prev[start+3]: %d, block[start]: %u, block[start+1]: %c, block[start+2]: %c, block[start+3]: %c\n",
    //              block_d[(i-1)*4+3], block_d[start], block_d[start+1], block_d[start+2], block_d[start+3]);
    // 00000000 01110010 00000000 00100000 prev1
    // 00000000 00000111 00000000 00000010 shr
    // 00000000 00000010 00000000 00000000 shl

    // 00000010 00000010 00000010 00000010
    // 11100111 10000011 11100111 11100111
    // 00000001 00000001 00000001 10111010

    //------------------------------------
    // 00000000 01000011 00000000 10101001 prev1
    // 00000000 00000100 00000000 00001010 shr
    // 00000000 00000011 00000000 00001001 shl

    // 00000010 00000010 00000010 10000000
    // 11100111 10000011 11100111 11001011
    // 00000001 00000001 00000001 10111010

}


__device__ __forceinline__
void permute_32(uint32_t current, uint32_t previous, uint32_t& prev1, uint32_t& prev2, uint32_t& prev3, uint64_t size, int total_padded_32){


    //
    // int prev = (i-1)*4;
    // int start = i*4;
    // uint32_t first = start<size ? block_d[start] : 0;
    // uint32_t second = start+1<size ? block_d[start+1] : 0;

    //
    uint32_t current = block_compressed_d[i];
    uint32_t previous =  i > 0 ? block_compressed_d[i-1] : 0;


    permutation_output_d[i] = (current << 16) | (previous >> 16);


    // uint64_t dist = ( ((uint64_t)current) << 32) | (uint64_t)previous;

    // prev1 = (uint32_t)(dist >> 3*8);
    // prev2 = (uint32_t)(dist >> 2*8);
    // prev3 = (uint32_t)(dist >> 1*8);
    // uint32_t test = i!=0 ?  block_d[prev+2] | (block_d[prev+3] << 8) | (first << 16) | (second << 24) : //
    //                                 0 | 0 << 8 | (first << 16) | (second << 24);
    // if(test != permutation_output_d[i]) printf("THERE IS AN ERROR IN permute_32 !!!\n");

}

__global__
void align_right(uint32_t* permutation_output_d, uint32_t* block_compressed_d, uint32_t* prev1_d, uint32_t* prev2_d, uint32_t* prev3_d,
                 uint64_t size, int total_padded_32, int shift){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //
        // int prev = (i-1)*4;
        // int start = i*4;

        // uint32_t first = start<size ? block_d[start] : 0;
        // uint32_t second = start+1<size ? block_d[start+1] : 0;
        // uint32_t third = start+2<size ? block_d[start+2] : 0;
        // uint32_t fourth = start+3<size ? block_d[start+3] : 0;

        // uint32_t prev_first = i > 0 ? (block_d[prev]) : 0;
        // uint32_t prev_second = i > 0 ? (block_d[prev+1]) : 0;
        // uint32_t prev_third = i > 0 ? (block_d[prev+2]) : 0;
        // uint32_t prev_fourth = i > 0 ? (block_d[prev+3]) : 0;

        //

        uint32_t current = block_compressed_d[i];
        uint32_t previous = i > 0 ? block_compressed_d[i-1] : 0;

        // uint32_t prev_chars = i> 0 ? ((prev_fourth << 24) | (prev_third << 16) | (prev_second << 8) | prev_first) : 0;
        // uint32_t current_chars = ((fourth << 24) | (third << 16) | (second << 8) | (first));
        //if(current != current_chars) printf("there IS AN ERROR IN align_right >>>> !!! %lu, %lu\n", current, current_chars);
        //if(previous != prev_chars) printf("THERE IS AN ERROR IN align_right ____!!! %lu, %lu\n", previous, prev_chars);

        uint64_t dist = ( ((uint64_t)current) << 32) | (uint64_t)previous;

        // uint64_t test = i > 0 ? (uint64_t)((((uint64_t)current_chars) << 32) | 
        //                  ((uint64_t)(prev_chars)))
        //                 : ((((uint64_t)current_chars) << 32) |
        //                 (0));

        // if(i ==8880) printf("%ld : THERE IS AN ERROR IN align_right !!! %d, %lu, %lu\n", i, previous==prev_chars , test, dist);
        // uint32_t dist1 = (((uint32_t)block_d[start] << 16 | (uint32_t)block_d[start+1] << 24)
        // | (permutation_output_d[i] & 0x0000ffff)) >> shift*8; // ((uint32_t)block_d[start] << 24 | (uint32_t)block_d[start+1] << 16) original
        // uint32_t dist2 = (((uint32_t)block_d[start+2] << 16 | (uint32_t)block_d[start+3] << 24) 
        // | ((permutation_output_d[i] & 0xffff0000) >> 16)) >> shift*8; // ((uint32_t)block_d[start+2] << 24 | (uint32_t)block_d[start+3] << 16) original
        // current_char : 01110010010000110010000010101001
        //                01100001001000001001100110000000
        // prev_chars :   11000010001000000110000101110100
        //                11100010011101000110100101100110

        // 11111111111111111111111111111111 11000010001000000110000101110100
        // 01110010010000110010000010101001 11000010001000000110000101110100    
    
        prev1_d[i] = (uint32_t)(dist >> shift*8);
        prev2_d[i] = (uint32_t)(dist >> (shift-1)*8);
        prev3_d[i] = (uint32_t)(dist >> (shift-2)*8);

    }

}


__global__ 
void is_incomplete(uint8_t* block, uint32_t* prev_incomplete_d, uint32_t* block_compressed_d, uint32_t* is_ascii_d, uint64_t size, int total_padded_32){
    static const uint32_t max_val = (uint32_t)(0b11000000u-1 << 24) | (uint32_t)(0b11100000u-1 << 16) | (uint32_t)(0b11110000u-1 << 8) | (uint32_t)(255);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //int prev = (i-1)*4;
        //printf("prev: %ld\n", i);
        int start = i*4;
        uint32_t first = start<size ? (uint32_t) block[start] : 0;
        uint32_t second = (start+1)<size ? (uint32_t) block[start+1] : 0;
        uint32_t third = (start+2)<size ? (uint32_t) block[start+2] : 0;
        uint32_t fourth = (start+3)<size ? (uint32_t) block[start+3] : 0;

        uint32_t val = first | second << 8 | third << 16 | fourth << 24;
        block_compressed_d[i] = val;
        prev_incomplete_d[i] = __vsubus4(val, max_val);
        is_ascii_d[i] =  ((val & 0x80808080) == 0);

        //printf("now %ld\n", i);
    }
}

__global__
void check_incomplete_ascii(uint32_t* block_compressed_d, uint32_t* prev_incomplete_d, uint32_t* is_ascii_d, uint32_t* error_d, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //
        // int prev = (i-1)*4;
        // int start = i*4;

        // uint32_t first = start<size ? (uint32_t) block_d[start] : 0;
        // uint32_t second = (start+1)<size ? (uint32_t) block_d[start+1] : 0;
        // uint32_t third = (start+2)<size ? (uint32_t) block_d[start+2] : 0;
        // uint32_t fourth = (start+3)<size ? (uint32_t) block_d[start+3] : 0;
        //

        //is_ascii_d[i] =

        // uint32_t test = (uint32_t)(((first&0b10000000) >> 7) //
        //             | ((second&0b10000000) >> 6)
        //             | ((third&0b10000000) >> 5)
        //             | ((fourth&0b10000000) >> 4)) == 0;
        // if(test != is_ascii_d[i]) printf("THERE IS AN ERROR IN check_incomlpete_ascii !!!\n");

        //else {is_ascii_d[i] = 0;}
        if(is_ascii_d[i] && i!=0) {error_d[i] = prev_incomplete_d[i-1];}
        else{
            uint32_t current = block_compressed_d[i];
            uint32_t previous = block_compressed_d[i-1];
            uint32_t prev1, prev2, prev3;
            uint32_t sc;
            uint32_t must32_80_sc;
            prev(current, previous, prev1, prev2, prev3, size, total_padded_32);
            check_special_cases(block_compressed_d[i], prev1, sc, size, total_padded_32);
            must_be_2_3_continuation_parallel_and_parallel_xor(prev1, prev2, sc, error_d[i], size, total_padded_32);
        }
    }

}


inline uint8_t prefix_or(uint32_t* is_ascii_d, uint64_t size, int total_padded_32){

    int error = thrust::reduce(thrust::cuda::par, is_ascii_d, is_ascii_d + total_padded_32);
    //cudaMemcpyAsync(&error, is_ascii_d+total_padded_32-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("error %d\n", error);

    return (uint8_t)error;
}
  

inline bool UTF8Validate(uint8_t * block_d, uint64_t size){
    int total_padded_32 = ((size + 3)/4) ;
    uint32_t* is_ascii_d;
    //uint32_t* permutation_output_d;
    uint32_t* prev_incomplete_d;
    uint32_t* block_compressed_d;
    uint32_t* error_d;
    int numBlock = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    // cudaEventRecord(gpu_start);
    //cudaEventSynchronize(gpu_stop);
    //cudaEventElapsedTime(&utf_runtime, gpu_start, gpu_stop);

    float check_special_cases_runtime = 0;


    //print8_d<uint8_t>(block_d, (int)size, ROW1);
    cudaMallocAsync(&is_ascii_d, sizeof(uint32_t)*total_padded_32, 0);
    cudaMallocAsync(&prev_incomplete_d, sizeof(uint32_t)*total_padded_32, 0);
    cudaMallocAsync(&block_compressed_d, sizeof(uint32_t)*total_padded_32, 0);
    cudaMallocAsync(&error_d, sizeof(uint32_t)*total_padded_32, 0);

    // cudaEventRecord(gpu_start);
    cudaMemsetAsync(error_d, 0, sizeof(uint32_t)*total_padded_32, 0);
    is_incomplete<<<numBlock, BLOCKSIZE>>>(block_d, prev_incomplete_d, block_compressed_d, is_ascii_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("special: %f\n", check_special_cases_runtime);

    // cudaEventRecord(gpu_start);
    // cudaMemsetAsync(is_ascii_d, 0, sizeof(uint32_t)*total_padded_32, 0);
    check_incomplete_ascii<<<numBlock, BLOCKSIZE>>>(block_compressed_d, prev_incomplete_d, is_ascii_d, error_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("check_incomplete: %f\n", check_special_cases_runtime);
    //print_d(is_ascii_d, total_padded_32, ROW1);



    uint8_t error = prefix_or(error_d, size, total_padded_32);

    //printf("error %d\n", error);

    //std::cout << "HERE" << std::endl;

    if(error != 0){ printf("Incomplete ASCII!\n"); return false;}

    //cudaMallocAsync(&permutation_output_d, sizeof(uint32_t)*total_padded_32,0);
    //uint32_t* prev1_d=is_ascii_d, *sc_d;
    //uint32_t* prev2_d=prev_incomplete_d, *prev3_d;
    //cudaMallocAsync(&prev2_d, sizeof(uint32_t)*total_padded_32, 0); // cudaMallocAsync(&prev2_d, sizeof(uint32_t)*total_padded_32, 0,0);
    //cudaMallocAsync(&prev3_d, sizeof(uint32_t)*total_padded_32,0);
    // cudaEventRecord(gpu_start);
    //permute_32<<<numBlock, BLOCKSIZE>>>(block_compressed_d, permutation_output_d, size, total_padded_32);
    //cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("permute: %f\n", check_special_cases_runtime);
    //print_d(permutation_output_d, total_padded_32, ROW1);


    //cudaMallocAsync(&prev1, sizeof(uint32_t)*total_padded_32,0);
    //cudaMallocAsync(&sc, sizeof(uint32_t)*total_padded_32,0);

    // cudaEventRecord(gpu_start);
    // align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_compressed_d, prev1_d, prev2_d, prev3_d, size, total_padded_32, 4-1);
    //cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("align_r: %f\n", check_special_cases_runtime);
    //printf("align1\n");
    //print_d(prev1_d, total_padded_32, ROW1);
    //cudaFreeAsync(permutation_output_d,0);
    //sc_d;
    //cudaMallocAsync(&sc_d, sizeof(uint32_t)*total_padded_32,0);
    // cudaEventRecord(gpu_start);
    //check_special_cases<<<numBlock, BLOCKSIZE>>>(block_compressed_d, prev1_d, sc_d, size, total_padded_32);
    //cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("special: %f\n", check_special_cases_runtime);
    //print_d(sc_d, total_padded_32, ROW1);

    //cudaFreeAsync(prev1_d,0);
    //cudaFreeAsync(block_compressed_d, 0);

    // align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev2_d, size, total_padded_32, 4-2);
    // cudaStreamSynchronize(0);
    //printf("align2\n");
    //print_d(prev2_d, total_padded_32, ROW1);

    // align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev3_d, size, total_padded_32, 4-3);
    // cudaStreamSynchronize(0);
    //printf("align3\n");
    //print_d(prev3_d, total_padded_32, ROW1);

    // uint32_t* must32_d;
    // cudaMallocAsync(&must32_d, sizeof(uint32_t)*total_padded_32,0);
    //uint32_t* must32_80_sc_d = prev1_d;
    //cudaMallocAsync(&must32_80_sc_d, sizeof(uint32_t)*total_padded_32,0);

    // cudaEventRecord(gpu_start);
    //must_be_2_3_continuation_parallel_and_parallel_xor<<<numBlock, BLOCKSIZE>>>(prev2_d, prev3_d, sc_d, must32_80_sc_d, size, total_padded_32);
    //cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("must_be_2_3: %f\n", check_special_cases_runtime);
    //printf("must32_d\n");
    //print_d(must32_d, total_padded_32, ROW1);
    //cudaFreeAsync(prev2_d,0);
    //cudaFreeAsync(prev3_d,0);

    // uint32_t* must32_80_d, *must32_80_sc_d;
    // cudaMallocAsync(&must32_80_d, sizeof(uint32_t)*total_padded_32,0);
    //cudaFreeAsync(sc_d,0);

    // cudaEventRecord(gpu_start);
    // single_parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(must32_d, 0x80, must32_80_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("special: %f\n", check_special_cases_runtime);
    //printf("must32_80_d\n");
    //print_d(must32_80_d, total_padded_32, ROW1);
    // cudaFreeAsync(must32_d,0);
    //uint32_t test_value;
    //cudaMemcpyAsync(&test_value, must32_80_d+8880, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("must32_80: %u\n", test_value);
    //cudaMemcpyAsync(&test_value, sc_d+8880, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("sc_d: %d\n", test_value);

    // cudaEventRecord(gpu_start);
    // parallel_xor<<<numBlock, BLOCKSIZE>>>(must32_80_d, sc_d, must32_80_sc_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("special: %f\n", check_special_cases_runtime);
    //cudaMemcpyAsync(&test_value, must32_80_sc_d+8880, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("must32_80_sc_d: %u\n", test_value);


    // cudaFreeAsync(must32_80_d,0);

    //error = prefix_or(must32_80_sc_d, size, total_padded_32);
    //cudaFreeAsync(must32_80_sc_d,0);
    return !(error);

}

__global__
void parallel_or_not_not_and_and_shift_right_shift_left_or_not_and(uint32_t* op_d, uint32_t* whitespace_d, uint32_t* in_string_d, uint32_t* follows_nonquote_scalar_d, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride){
        uint32_t op = op_d[i];
        uint32_t whitespace = whitespace_d[i];
        uint32_t in_string = in_string_d[i];

        uint32_t scalar = op | whitespace;
        scalar = ~scalar;
        in_string = ~in_string;
        uint32_t nonquote_scalar = scalar & in_string;
        in_string = in_string & op;
        uint32_t overflow = nonquote_scalar >> 31;
        uint32_t follows_nonquote_scalar = nonquote_scalar << 1;
        follows_nonquote_scalar = follows_nonquote_scalar | overflow;
        in_string_d[i] = in_string;
        follows_nonquote_scalar_d[i] = follows_nonquote_scalar;
    }

}

__global__
void get(uint8_t* block_d, uint32_t* output1, uint32_t* output2, uint64_t size, int total_padded_32){

    // uint8_t table1_g[16] = {      
    //     0, 0, 1, 0,
    //     0, 1, 0, 0,
    //     0, 0, 0, 0,
    //     0, 0, 0, 0
    // };
    // uint8_t table2_g[16] = {
    //     0, 0, 1, 0,
    //     0, 0, 0, 0,
    //     0, 0, 0, 0,
    //     1, 0, 0, 0
    // };
    
    // int tid = threadIdx.x;
    // __shared__ uint8_t table1[16];
    // __shared__ uint8_t table2[16];
    // //tid < 16 ? table1[tid] = table1_g[tid] : NULL;
    // //tid < 16 ? table2[tid] = table2_g[tid] : NULL;
    // //tid < 16 ? table3[tid] = table3_g[tid] : NULL;
    // if(tid == 0){
    //     for(int k=0; k<16; k++){
    //         table1[k] = table1_g[k];
    //         table2[k] = table2_g[k];
    //     }
    // }
    // __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        
        int start = i*32;
        uint32_t res1 = 0;
        uint32_t res2 = 0;
        // uint32_t res3 = 0;
        // uint32_t res4 = 0;

        for (int j = start; j<start+32 && j<size; j++){

            uint8_t block = block_d[j];
            // uint8_t block_low = block & 0x08;
            // uint8_t block_high = block >> 4;
            // if (table1[block_high] & table2[block_low] == 0) continue;
            block == '\\' ? res1 |= 1 << (j-start) : NULL;

            block == '\"' ? res2 |= 1 << (j-start) : NULL;

            // block == '[' || '{' || ']' || '}' || ':' || ',' ? res3 |= 1 << (j-start) :  NULL;

            // block == '\t' || '\r' ? res4 |= 1 << (j-start) : NULL;

        }
        output1[i] = res1;
        output2[i] = res2;
        // output3[i] = res3;
        // output4[i] = res4;
    }
}

__global__
void split(uint32_t* input, uint8_t* output, uint64_t size, int total_padded_32){
    // uint8_t val_arr[2] = {0x00, 0xff};
    // int tid = threadIdx.x;
    // int index = blockIdx.x * blockDim.x + tid;
    // int stride = blockDim.x * gridDim.x;
    // for(long i = index; i< total_padded_32; i+=stride)
    // {
    //     uint32_t val = input[i];
    //     int start = i*32;
    //     for(int j=0; j<32 && j+start<size; j++){
    //         output[start+j] = val_arr[((val >> (j)) & 0b00000001)];
    //     }
    // }
    static uint8_t val_arr[2] = {0x00, 0xff};
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< size; i+=stride)
    {
        int j = (i >> 5);
        int reminder = i % 32;
        uint32_t val = input[j];
        output[i] = val_arr[((val >> (reminder)) & 0b00000001)];
    }
}

__global__
void compact(uint8_t* input, uint32_t* output, uint64_t size, int total_padded_32){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int start = i*32;
        uint32_t res = 0;
        for (int j = start; j<start+32; j++){
            res |= (input[j]<< (32-(j-start))) ;
        }
        output[i] = res;
    }
}

__global__
void sum_ones(uint32_t* real_quote_d, uint32_t* prediction_d, uint32_t* total_one_32_d,  uint64_t size, int total_padded_32, int total_padded_32_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32_32; i+=stride)
    {
        int start  = i*32;
        int end = (i+1)*32;
        int total = 0;
        for(int j = start; j< end && j<total_padded_32; j++){
            int total_one = __popc(real_quote_d[j]);
            total += total_one;
            prediction_d[j] = (uint32_t) total_one;
        }
        total_one_32_d[i] = total;
    }
}

__global__
void scatter(uint32_t* total_one_32_d, uint32_t* total_one_d, uint32_t* prefix_sum_ones, int size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< size; i+=stride)
    {
        int start = i*32;
        int end = (i+1)*32;
        uint32_t current_value = total_one_32_d[i];
        for(int j = start; j<end && j<total_padded_32; j++){
            prefix_sum_ones[j] = current_value;
            current_value += total_one_d[j];

        }
    }
}

__global__
void scatter_block(uint32_t* total_one_d, uint32_t* one_d, uint32_t* prefix_sum_ones, int size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< size; i+=stride)
    {
        int start = i*32;
        int end = (i+1)*32;
        uint32_t current_value = total_one_d[i];
        for(unsigned int j = 0; j<32 && (start+j)<total_padded_32 ; j++){

            current_value += (uint32_t)((one_d[i] >> j) & 1);

            prefix_sum_ones[start+j] = current_value;
            
        }
    }
}

__global__
void fact(uint32_t* real_quote_d, uint32_t* prefix_sum_ones, uint32_t* res, int size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride)
    {

        bool overflow_b = prefix_sum_ones[i] & 1;
        uint32_t shl = real_quote_d[i] << 1;
        uint32_t clear = 1;
        res[i] = real_quote_d[i];
        for(int j=0; j<32; j++){
            shl = res[i] << 1;
            res[i] = res[i] ^ (shl & (clear << (j)));
        }
        //if (i == 1) printf("%d\n", res[i]);
        res[i] = overflow_b == 1 ? ~res[i] : res[i];
    }
}

__global__
void not_zero_index(uint32_t* input, uint32_t* output, int sum, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        if(input[i]>0){
            int j=0;
            while(j<sum && input[i] != output[j]) j++; 
            output[j+sum] = input[i+total_padded_32];
        }
    }
}

__global__
void predict(uint8_t* block_d,  uint32_t* real_quote_d, uint32_t* prediction_d, uint64_t size, int total_padded_32){
    enum hypothesis_val hypothesis = in;
    int run = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int start = i*32;
        uint32_t cur_q = real_quote_d[i];
        while(run < 2){ //TODO last item, number exponent
            enum hypothesis_val current_state = hypothesis;
            int state = 0;
            int j = 0;
            for(j = 0; j<32 && start+j < size; j++){
                uint8_t cur_b = block_d[start+j];
                if(current_state == in && ((cur_q >> j) & 1) == 1) current_state = out;
                else if(current_state == in) continue;
                else if(current_state == out){
                    switch(state){
                        case 0:{
                            if(((cur_q >> j) & 1) == 1) {
                                state = 0;
                                current_state = in;
                            }
                            else if(cur_b=='t') state = 1;
                            else if(cur_b=='r') state = 2;
                            else if(cur_b=='u') state = 15;
                            else if(cur_b=='e') state = 4;
                            else if(cur_b=='f') state = 5;
                            else if(cur_b=='a') state = 6;
                            else if(cur_b=='l') state = 16;
                            else if(cur_b=='s') state = 8;
                            else if(cur_b=='n') state = 9;
                            else if(cur_b=='.') state = 13;
                            else if((cur_b>= 0x30 && cur_b <= 0x39) || cur_b == 0x2d) state = 12;
                            else if(cur_b == '[' ||  cur_b == '{' || cur_b == ' ' || cur_b == '\n' || cur_b == '\t' || cur_b == '\r' || cur_b == ':' || cur_b == ',') state = 0;
                            else if((cur_b == ']' || cur_b == '}') && j+start < size-1 ) state = 4;
                            else if((cur_b == ']' || cur_b == '}') && j+start == size-1 ) {state = 0; j=40;}

                            else state = 20;
                        break;}
                        case 1:{
                            state = cur_b == 'r' ? 2 : 20; 
                        break;}
                        case 2:{
                            state = cur_b == 'u' ? 3 : 20;
                        break;}
                        case 3:{
                            state = cur_b == 'e' ? 4 : 20;
                        break;}
                        case 4:{
                            if(cur_b == ' ' || cur_b == '\n' || cur_b == '\t' || cur_b == '\r') state=4;
                            else if(cur_b == ',' || cur_b == '}' || cur_b == ']') state = 0;
                            else state = 20;
                        break;}
                        case 5:{
                            state = cur_b == 'a' ? 6 : 20;
                        break;}
                        case 6:{
                            state = cur_b == 'l' ? 7 : 20;
                        break;}
                        case 7:{
                            state = cur_b == 's' ? 8 : 20;
                        break;}
                        case 8:{
                            state = cur_b == 'e' ? 4 : 20;
                        break;}
                        case 9:{
                            state = cur_b == 'u' ? 10 : 20;
                        break;}
                        case 10:{
                            state = cur_b == 'l' ? 11 : 20;
                        break;}
                        case 11:{
                            state = cur_b == 'l' ? 4 : 20;
                        break;}
                        case 12:{
                            if(cur_b>= 0x30 && cur_b <= 0x39) state = 12;
                            else if(cur_b == 0x2e) state = 13;
                            else if(cur_b == 0x65) state = 14;
                            else if(cur_b == '\n' || cur_b == '\t' || cur_b == '\r') state = 4;
                            else if(cur_b == ',' || cur_b == '}' || cur_b == ']') state = 0;
                            else state = 20;
                        break;}
                        case 13:{
                            if(cur_b>= 0x30 && cur_b <= 0x39) state = 13;
                            else if(cur_b == 0x65) state = 14;
                            else if(cur_b == '\n' || cur_b == '\t' || cur_b == '\r') state = 4;
                            else if(cur_b == ',' || cur_b == '}' || cur_b == ']') state = 0;
                            else state = 20;
                        break;}
                        case 14:{
                            if(cur_b>= 0x30 && cur_b <= 0x39) state = 14;
                            else if(cur_b == '\n' || cur_b == '\t' || cur_b == '\r') state = 4;
                            else if(cur_b == ',' || cur_b == '}' || cur_b == ']') state = 0;
                            else state = 20;
                        break;}
                        case 15:{
                            if(cur_b == 'e') state = 4;
                            else if(cur_b == 'l') state = 11;
                            else state = 20;
                        break;}
                        case 16:{
                            if(cur_b == 'l') state = 4;
                            else if(cur_b == 's') state = 8;
                            else state = 20;
                            
                        break;}
                        case 20:{
                            hypothesis = hypothesis == in ? out : in;
                            j = 40;
                        break;}
                    }
                }
                else break;
            }
            run++;
            if (run>1 && j==33) {hypothesis = unknown; break;}
            else if (run>1) break;
            hypothesis = out;
        }
        prediction_d[i] = 0;
        for(int j = 0; j<32; j++){
            if(hypothesis == in) prediction_d[i] |= (1<<j);
            if(((cur_q >> j) & 1) == 1) hypothesis =  (hypothesis == in) ? out : in;
        }

    }
}


__global__
void classify(uint8_t* block_d, uint32_t* op_d, uint32_t* whitespace_d, uint64_t size, int total_padded_32){
    // uint8_t ascii_table1_g[256] = {
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    // };
    // uint8_t ascii_table2_g[256] = {
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    // };
    int tid = threadIdx.x;
    // __shared__ uint8_t ascii_table1[256];
    // __shared__ uint8_t ascii_table2[256];
    // if(tid == 0){
    //     for(int k=0; k<256; k++){
    //         ascii_table1[k] = ascii_table1_g[k];
    //         ascii_table2[k] = ascii_table2_g[k];
    //     }
    //     // printf("FDFDFDFD\n");
    // }
    // __syncthreads();

    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int start = i*32;
        uint32_t res_op = 0;
        uint32_t res_wt = 0;
        for (int j = start; j<start+32 && j<size; j++){
            uint8_t block = block_d[j];
            uint8_t block_low = block & 0x08;
            if (block_low != 8) continue;
            // res_op |= ((uint32_t)ascii_table1[block_d[j]] << (j-start));
            // res_wt |= ((uint32_t)ascii_table2[block_d[j]] << (j-start));
            res_op |= (((block == '{' ||
                    block == '[' ||
                    block == '}' ||
                    block == ']' ||
                    block == ':' ||
                    block == ','
                    ) ? 1 : 0) << (j-start)) ;
            res_wt |= (((block == ' ' ||
                    block == '\t' ||
                    //block_d[j] == '\n' ||
                    block == '\r'
                    ) ? 1 : 0) << (j-start)) ;
        }
        op_d[i] = res_op;
        whitespace_d[i] = res_wt;
    }

}

__global__
void find_escaped(uint32_t* backslashes_d, uint32_t* escaped_d, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        uint32_t has_overflow = 2;
        uint32_t even_bits = 0x55555555UL;
        long j=i-1;
        if(i==0) has_overflow = 0;
        while(has_overflow == 2){
            uint32_t j_backslash = backslashes_d[j];
            uint32_t follows_escape_t = j_backslash << 1;
            uint32_t odd_seq_t = j_backslash & ~even_bits & ~follows_escape_t;
            uint32_t last_zero = ~(j_backslash | odd_seq_t);
            uint32_t last_one = j_backslash & odd_seq_t;
            //if(i==882) printf("%u\n", backslashes_d[j]);

            //if(i==882) printf("%u\n", last_zero);
            //if(i==882) printf("%u\n", last_one);

            uint32_t last_two_bits = (j_backslash & 0xC0000000UL) >> 30;
            //if(i==882) printf("%d\n", last_two_bits);
            
            //if(i==882) printf("%d\n", (last_two_bits == 3));

            has_overflow = (last_two_bits == 2 || (last_two_bits == 3 && last_one>last_zero)) ? 1 : ((last_two_bits == 3 && last_one==last_zero) ? 2 : 0);
            //if(i==882) printf("%u\n", (last_two_bits == 3));
            j--;
        }
        uint32_t backslashes = backslashes_d[i];
        backslashes &= ~has_overflow;
        uint32_t follows_escape = (backslashes << 1) | has_overflow;
        uint32_t odd_seq = backslashes & ~even_bits & ~follows_escape;
        uint32_t sequence_starting_even_bits = odd_seq + backslashes;
        uint32_t invert_mask = sequence_starting_even_bits << 1;
        escaped_d[i] = (even_bits ^ invert_mask) & follows_escape;
        
    }
}

__global__
void assign_open_close(uint8_t* block_d, uint32_t* open_d, uint32_t* close_d, int size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int start = i*32;
        uint32_t res_open = 0;
        uint32_t res_close = 0;
        for (int j = start; j<start+32 && j<size; j++){
            res_open |= (((block_d[j] == '{' ||
                    block_d[j] == '[') ? 1 : 0) << (j-start)) ;
            res_close |= (((block_d[j] == '}' ||
                    block_d[j] == ']') ? 1 : 0) << (j-start)) ;
        }
        open_d[i] = res_open;
        close_d[i] = res_close;
    }
}

__global__
void parallel_copy(uint8_t* tokens_d, uint8_t* res_d, uint32_t res_size, uint32_t last_index_tokens){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(uint32_t i = index; i< last_index_tokens; i+=stride)
    {
        //if (i==192)printf("complete_d: %d\n", *(complete_records_d+open_close_reduced_size_tokens-1));
        //if (i==64)printf("complete_d: %d\n", *(complete_records_d+open_close_reduced_size_tokens-1));

        /*    
        if(i==0){
            res_d[0] = '{';
            res_d[res_size-2] = '}';
            res_d[res_size-1] = ','; 
        }
        else{
            uint32_t start = i==1 ? 0 : complete_records_d[i-2]+1;
            uint32_t res_start = i==1 ? 1 : complete_records_d[i-2]+3+(i-2);
            uint32_t res_end = i==1 ? complete_records_d[i-1]+1 : complete_records_d[i-1]+2+(i-2);
            uint32_t end = complete_records_d[i-1];
            //for(uint32_t j=0; j<end+1-start; j++){
            //    res_d[res_start+j] = tokens_d[start+j];
            //}
            uint32_t* j=thrust::lower_bound(thrust::device, complete_records_d, complete_records_d+open_close_reduced_size_tokens, i);
            uint32_t  = (uint32_t) j - (uint32_t) complete_records_d;
            cudaMemcpyAsync(res_d+res_start, tokens_d+start , sizeof(uint8_t)*(end+1-start), cudaMemcpyDeviceToDevice, 0);
            //thrust::copy(thrust::device, &tokens_d[start], &tokens_d[end+1], &res_d[res_start]);
            if (i < open_close_reduced_size_tokens) res_d[res_end+1] = ',';
            //printf("endc: %c\n", res_d[end]);

        }
        */
        /*
        uint32_t* j_pointer;
        //thrust::device_ptr<uint32_t> thrust_complete_records_d = thrust::device_pointer_cast(complete_records_d);

        //printf("%d\n", open_close_reduced_size_tokens);
        j_pointer =  thrust::lower_bound(thrust::seq,
            complete_records_d,
            complete_records_d+open_close_reduced_size_tokens-1,
            i);
        
        ptrdiff_t j_index = (j_pointer - complete_records_d);

        //printf("i: %d, *j_pointer: %d, j_pointer: %p, complete_records: %p, j_index: %ld,\n",
        //        i,
        //        j_pointer[0],
        //        j_pointer,
        //        complete_records_d,
        //        j_index);

            
        uint32_t j = j_index == 0 ? i : i - (*(j_pointer-1)+1);
        uint32_t start = j_index==0 ? 0 : *(j_pointer-1);
        uint32_t res_start = j_index==0 ? 1 : *(j_pointer-1)+j_index + 2;
        res_d[res_start+j] = tokens_d[i];
        if(i<last_index_tokens-1 && res_start+j < res_size-4 && tokens_d[i] == '\n') res_d[res_start+j] = ','; //why shifted twice?
        */
        if(i == 0){
            res_d[0] = '[';
            res_d[res_size-2] = ']';
            res_d[res_size-1] = ',';
        }
        res_d[i+1] = tokens_d[i];
        if(tokens_d[i] == '\n' & i < last_index_tokens - 1) res_d[i+1] = ',';
        

        
    }
}

__global__
void count_set_bits(uint32_t* input, uint32_t* total_bits, uint32_t size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(uint32_t i = index; i< size; i+=stride){
        int total = __popc(input[i]);
        total_bits[i] = (uint32_t) total;
    }
}

__global__
void remove_and_copy(uint32_t* set_bit_count, uint32_t* in_string, uint8_t* block_d, uint8_t* in_string_8_d, uint32_t size, uint32_t total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(uint32_t i = index; i< total_padded_32; i+=stride){
        uint32_t total_before = i > 0 ? set_bit_count[i-1] : 0;
        uint32_t current_total = 0;
        for(int j = 0; j<32 && i*32+j<size; j++){
            uint8_t current_bit = (in_string[i] >> j) & 1;
            current_bit == 1 ? in_string_8_d[total_before+current_total] = block_d[i*32+j] : NULL;
            current_total += current_bit;

        }  
    }
}

inline uint8_t * Tokenize(uint8_t* block_d, uint64_t size, int &ret_size, uint32_t &last_index_tokens){
    int total_padded_32 = (size+31)/32 ;
    int numBlock = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlockBySize = (size + BLOCKSIZE - 1) / BLOCKSIZE;
    uint32_t* backslashes_d;
    //print8_d<uint8_t>(block_d, size, ROW1);

    uint32_t* quote_d;

    uint32_t* general_ptr;

    cudaMallocAsync(&general_ptr, total_padded_32*sizeof(uint32_t)*ROW4, 0);

    clock_t start, end;

    quote_d = general_ptr;
    backslashes_d = general_ptr+total_padded_32;
    //whitespace_d = general_ptr+total_padded_32*ROW4;
    //op_d = general_ptr+total_padded_32*ROW5;

    start = clock();
    //cudaMallocAsync(&backslashes_d, total_padded_32 * sizeof(uint32_t),0);
    //cudaMallocAsync(&quote_d, total_padded_32 * sizeof(uint32_t),0);
    get<<<numBlock, BLOCKSIZE>>>(block_d, backslashes_d, quote_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(backslashes_d, total_padded_32, ROW1);
    //printf("%d\n", total_padded_32);
    //print_d(quote_d, total_padded_32, ROW1);


    uint32_t* escaped_d;
    escaped_d = general_ptr+total_padded_32*ROW2;
    start = clock();
    //cudaMallocAsync(&escaped_d, total_padded_32 * sizeof(uint32_t),0);
    find_escaped<<<numBlock, BLOCKSIZE>>>(backslashes_d, escaped_d, total_padded_32);
    cudaStreamSynchronize(0);
    //cudaFreeAsync(backslashes_d,0);
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    // printf("ecaped backslashes: \n");
    // print_d(escaped_d, total_padded_32, ROW1);

    // uint32_t* quote_d;
    // start = clock();
    // cudaMallocAsync(&quote_d, total_padded_32 * sizeof(uint32_t),0);
    // get<<<numBlock, BLOCKSIZE>>>(block_d, quote_d, size, total_padded_32, 1);//////////////
    // cudaStreamSynchronize(0);
    // end = clock();
    // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    // print_d(quote_d, total_padded_32, ROW1);

    uint32_t* real_quote_d;
    real_quote_d = general_ptr+total_padded_32;
    start = clock();
    //cudaMallocAsync(&real_quote_d, total_padded_32 * sizeof(uint32_t),0);
    parallel_not<<<numBlock, BLOCKSIZE>>>(escaped_d, escaped_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(escaped_d, total_padded_32, ROW1);
    start = clock();
    parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(quote_d, escaped_d, real_quote_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    end = clock();
    // cudaFreeAsync(quote_d,0);
    // cudaFreeAsync(escaped_d,0);
    // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(real_quote_d, total_padded_32, ROW1);
    /*
    start = clock();
    uint32_t* prediction_d;
    cudaMallocAsync(&prediction_d, total_padded_32 * sizeof(uint32_t),0);
    predict<<<numBlock, BLOCKSIZE>>>(block_d, real_quote_d, prediction_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    cudaFreeAsync(real_quote_d,0);
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    */

    //print_d(prediction_d, total_padded_32, ROW1);

    
    uint32_t* in_string_d;
    in_string_d = general_ptr;
    //cudaMallocAsync(&in_string_d, total_padded_32 * sizeof(uint32_t),0);
    
    uint32_t* total_one_d;
    total_one_d = general_ptr+total_padded_32*ROW2;
    uint32_t* total_one_32_d;
    start = clock();
    int total_padded_32_div_32 = (total_padded_32+31)/32 ;

    //cudaMallocAsync(&total_one_d, total_padded_32*sizeof(uint32_t),0);
    cudaMallocAsync(&total_one_32_d, (total_padded_32_div_32)*sizeof(uint32_t),0);

    int smallNumBlock = (total_padded_32_div_32 + BLOCKSIZE - 1) / BLOCKSIZE;

    sum_ones<<<smallNumBlock, BLOCKSIZE>>>(real_quote_d, total_one_d, total_one_32_d,  size, total_padded_32, total_padded_32_div_32);
    cudaStreamSynchronize(0);
    //print_d(total_one_d, total_padded_32, ROW1);
    end = clock();
    //std::cout << "Time elapsed Sum ones: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    start = clock();
    thrust::exclusive_scan(thrust::cuda::par, total_one_32_d, total_one_32_d+(total_padded_32_div_32), total_one_32_d);
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(total_one_reduced_d, sum, ROW1);

    //printf("%d\n", 1234);

    uint32_t* prefix_sum_ones;
    prefix_sum_ones = general_ptr+total_padded_32*ROW3;
    //cudaMallocAsync(&prefix_sum_ones, total_padded_32*sizeof(uint32_t),0);
    start = clock();

    scatter<<<smallNumBlock, BLOCKSIZE>>>(total_one_32_d, total_one_d, prefix_sum_ones, total_padded_32_div_32, total_padded_32);
    cudaStreamSynchronize(0);

    end = clock();
    //print_d(total_one_d, total_padded_32, ROW1);
    //std::cout << "Time elapsed scatter: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;



    start = clock();
    fact<<<numBlock, BLOCKSIZE>>>(real_quote_d, prefix_sum_ones, in_string_d, total_padded_32_div_32, total_padded_32); // ROW1, ROW3, ROW0
    cudaStreamSynchronize(0);
    //cudaFreeAsync(total_one_d,0);
    cudaFreeAsync(total_one_32_d,0);
    //cudaFreeAsync(prefix_sum_ones,0);
    //cudaFreeAsync(real_quote_d,0);
    end = clock();

    //print_d(in_string_d, total_padded_32, ROW1);

    //std::cout << "Time elapsed Fact: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    /*
    cudaMemcpyAsync(in_string_d, prediction_d, sizeof(uint32_t)*total_padded_32, cudaMemcpyDeviceToDevice);
    cudaFreeAsync(prediction_d,0);
    */
    /////////*
    /*
    uint8_t* real_quote_8_d;
    cudaMallocAsync(&real_quote_8_d, size * sizeof(uint8_t),0);
    split<<<numBlock, BLOCKSIZE>>>(real_quote_d, real_quote_8_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    thrust::inclusive_scan(thrust::cuda::par, real_quote_8_d, real_quote_8_d + size, real_quote_8_d);
    single_parallel_and<uint8_t><<<numBlock, BLOCKSIZE>>>(real_quote_8_d, 0b00000001, real_quote_8_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    uint32_t* in_string_d;
    cudaMallocAsync(&in_string_d, total_padded_32 * sizeof(uint32_t),0);
    compact<<<numBlock, BLOCKSIZE>>>(real_quote_8_d, in_string_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    
    //////////

    ///////////////////////////////////////
    

    prefix_xor(real_quote_d, in_string_t, size, total_padded_32);
    uint32_t* last_bits_d;
    cudaMallocAsync(&last_bits_d, total_padded_32 * sizeof(uint32_t),0);
    get_last_bits<<<>>>(in_string_t, last_bits_d, total_padded_32);///////////////
    uint32_t* last_bit_propagate_d;
    cudaMallocAsync(&last_bit_propagate_d, total_padded_32 * sizeof(uint32_t),0);
    prefix_xor(last_bits_d, last_bit_propagate_d, size, total_padded_32);
    uint32_t* in_string_d;
    cudaMallocAsync(&in_string_d, total_padded_32*sizeof(uint32_t),0);
    prefix_xor(real_quote_d, in_string_d, size, total_padded_32);
    parallel_xor<<<>>>(in_string_d, last_bit_propagate_d, in_string_d, size, total_padded_32);
    */
    /////////////////////////////////////////////
    
    uint32_t *whitespace_d, *op_d;
    //uint32_t* general_ptr;
    uint32_t* follows_nonquote_scalar_d;
    //cudaMallocAsync(&general_ptr, total_padded_32*sizeof(uint32_t)*2,0);

    op_d = general_ptr+total_padded_32;
    follows_nonquote_scalar_d = general_ptr+total_padded_32*ROW2;
    whitespace_d = general_ptr+total_padded_32*ROW3;
    //cudaMallocAsync(&whitespace_d, total_padded_32*sizeof(uint32_t),0);
    //cudaMallocAsync(&op_d, total_padded_32*sizeof(uint32_t),0);
    start = clock();
    classify<<<numBlock, BLOCKSIZE>>>(block_d, op_d, whitespace_d, size, total_padded_32);////////////////////
    cudaStreamSynchronize(0);
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    //print_d(whitespace_d, total_padded_32, ROW1);
    //print_d(op_d, total_padded_32, ROW1);
    
    // op_d, whitespace_d, in_string_d, size, total_padded_32
    // follows_nonquote_scalar_d, in_string_d
    parallel_or_not_not_and_and_shift_right_shift_left_or_not_and<<<numBlock, BLOCKSIZE>>>(op_d, whitespace_d, in_string_d, follows_nonquote_scalar_d, size, total_padded_32);


    //////////////////////////////////////////////////////////////////////////
    // uint32_t* scalar_d;

    // scalar_d = general_ptr+total_padded_32;
    // start = clock();
    // //cudaMallocAsync(&scalar_d, total_padded_32*sizeof(uint32_t),0);
    // parallel_or<<<numBlock, BLOCKSIZE>>>(op_d, whitespace_d, scalar_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    
    //print_d(scalar_d, total_padded_32, ROW1);
    // start = clock();
    // parallel_not<<<numBlock, BLOCKSIZE>>>(scalar_d, scalar_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    
    //print_d(scalar_d, total_padded_32, ROW1);

    // uint32_t* nonquote_scalar_d;
    // nonquote_scalar_d = general_ptr+total_padded_32*2;
    // //cudaMallocAsync(&nonquote_scalar_d, total_padded_32*sizeof(uint32_t),0);
    // start = clock();
    // parallel_not<<<numBlock, BLOCKSIZE>>>(in_string_d, in_string_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    //print_d(in_string_d, total_padded_32, ROW1);
    
    // start = clock();
    // parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(scalar_d, in_string_d, nonquote_scalar_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    // start = clock();
    // parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(in_string_d, op_d, in_string_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();

    //print_d(nonquote_scalar_d, total_padded_32, ROW1);
    
    // uint32_t* overflow;
    // overflow = general_ptr+total_padded_32*3;
    // //cudaMallocAsync(&overflow, total_padded_32*sizeof(uint32_t),0);
    // start = clock();
    // parallel_shift_right<<<numBlock, BLOCKSIZE>>>(nonquote_scalar_d, overflow, 31, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    //print_d(overflow, total_padded_32, ROW1);
    
    // follows_nonquote_scalar_d = general_ptr+total_padded_32*4;
    // //cudaMallocAsync(&follows_nonquote_scalar_d, total_padded_32*sizeof(uint32_t),0);
    // start = clock();
    // parallel_shift_left<<<numBlock, BLOCKSIZE>>>(nonquote_scalar_d, follows_nonquote_scalar_d,  1, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(follows_nonquote_scalar_d, total_padded_32, ROW1);

    // start = clock();
    // parallel_or<<<numBlock, BLOCKSIZE>>>(follows_nonquote_scalar_d, overflow, follows_nonquote_scalar_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(follows_nonquote_scalar_d, total_padded_32, ROW1);

    //cudaFreeAsync(op_d,0);
    // cudaFreeAsync(follows_nonquote_scalar_d,0);                                                    //Remove later
    // cudaFreeAsync(scalar_d,0);
    // cudaFreeAsync(nonquote_scalar_d,0);
    // cudaFreeAsync(overflow,0);
    //cudaFreeAsync(general_ptr,0);

    // start = clock();
    // parallel_not<<<numBlock, BLOCKSIZE>>>(whitespace_d, whitespace_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    //print_d(whitespace_d, total_padded_32, ROW1);


    // start = clock();
    // parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(in_string_d, whitespace_d, in_string_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //cudaFreeAsync(whitespace_d,0);                                                         //// let it be here for now
    //print_d(in_string_d, total_padded_32, ROW1);
    ///////////////////////////////////////////////////////////////

    uint32_t* set_bit_count;
    set_bit_count = general_ptr+total_padded_32;
    //cudaMallocAsync(&set_bit_count, sizeof(uint32_t)*total_padded_32, 0);
    count_set_bits<<<numBlock, BLOCKSIZE>>>(in_string_d, set_bit_count, total_padded_32);
    cudaStreamSynchronize(0);


    thrust::inclusive_scan(thrust::cuda::par, set_bit_count, set_bit_count+total_padded_32, set_bit_count);

    uint8_t* in_string_8_d;
    cudaMallocAsync(&in_string_8_d, size * sizeof(uint8_t),0);

    remove_and_copy<<<numBlock, BLOCKSIZE>>>(set_bit_count, in_string_d, block_d, in_string_8_d, size, total_padded_32);
    cudaStreamSynchronize(0);

    cudaMemcpyAsync(&last_index_tokens, set_bit_count+total_padded_32-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //print8_d<uint8_t>(in_string_8_d, last_index_tokens, ROW1);

    //cudaFreeAsync(set_bit_count, 0);
    // start = clock();
    // split<<<numBlockBySize, BLOCKSIZE>>>(in_string_d, in_string_8_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    //cudaFreeAsync(in_string_d,0);
    //print_d(in_string_d, total_padded_32, ROW1);
    //print8_d<int>(in_string_8_d, size, ROW1);
    cudaFreeAsync(general_ptr,0);




    uint8_t* in_string_out_d;
    // start = clock();
    // parallel_and<uint8_t><<<numBlock, BLOCKSIZE>>>(in_string_8_d, block_d, in_string_8_d, size, size);
    // cudaStreamSynchronize(0);
    // end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //in_string_8  = (uint8_t*) malloc(size*sizeof(uint8_t));
    //cudaMemcpyAsync(in_string_8, in_string_8_d, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost);

    // uint8_t* filtered_string_8_d;
    // int sum = thrust::count_if(thrust::cuda::par, in_string_8_d, in_string_8_d+size, not_zero());
    // cudaMallocAsync(&filtered_string_8_d, sum*sizeof(uint8_t),0);

    // thrust::copy_if(thrust::cuda::par, in_string_8_d, in_string_8_d+size, filtered_string_8_d, not_zero());

    //uint8_t* in_string_out;
    //in_string_out = (uint8_t* )malloc(sum*sizeof(uint8_t));
    //in_string_out[0] = 'A';

    //cudaMemcpyAsync(in_string_out, filtered_string_8_d, sizeof(uint8_t), cudaMemcpyDeviceToHost);

    //printf("filtered: %c\n", in_string_out[0]);

    //print8(in_string_8, size, ROW1);

    //cudaMemcpyAsync(in_string_out, filtered_string_8_d, sizeof(uint8_t)*sum, cudaMemcpyDeviceToHost);


    
    //last_index_tokens = sum;
    //printf("last_index3: %d, open_close_size3: %d\n", line_length, total_line);
    //free(in_string_out);
    in_string_out_d = in_string_8_d;
    ret_size = last_index_tokens;
    //cudaFreeAsync(in_string_d,0);

    return in_string_out_d;
}


inline uint32_t* get_last_record(uint8_t* block_d, int size, uint32_t &last_index, uint32_t& open_close_reduced_size){
    clock_t start, end;
    double time = 0;
    
    int total_padded_32 = (size+31)/32 ;
    int numBlock = (size + BLOCKSIZE - 1) / BLOCKSIZE;
    int total_padded_32_div_32 = (total_padded_32+31)/32 ;
    int smallNumBlock = (total_padded_32_div_32 + BLOCKSIZE - 1) / BLOCKSIZE;

    uint32_t* general_ptr;
    cudaMallocAsync(&general_ptr, total_padded_32*sizeof(uint32_t)*6,0);

    uint32_t* open_d;
    uint32_t* close_d;

    uint32_t* open_count_d;
    uint32_t* close_count_d;

    uint32_t* open_count_32_d;
    uint32_t* close_count_32_d;

    uint32_t* open_prefix_sum_d;
    uint32_t* close_prefix_sum_d;

    open_d = general_ptr;
    close_d = general_ptr+total_padded_32;
    open_count_d = general_ptr+total_padded_32*2;
    close_count_d = general_ptr+total_padded_32*3;
    open_prefix_sum_d = general_ptr+total_padded_32*4;
    close_prefix_sum_d = general_ptr+total_padded_32*5;

    //uint8_t* block_d;


    //cudaMallocAsync(&block_d, size*sizeof(uint8_t),0);
    //cudaMemcpyAsync(block_d, block, size*sizeof(uint8_t), cudaMemcpyHostToDevice);

    //cudaMallocAsync(&open_d, total_padded_32*sizeof(uint32_t),0);
    //cudaMallocAsync(&close_d, total_padded_32*sizeof(uint32_t),0);

    //cudaMallocAsync(&open_count_d, total_padded_32*sizeof(uint32_t),0);
    //cudaMallocAsync(&close_count_d, total_padded_32*sizeof(uint32_t),0);

    cudaMallocAsync(&open_count_32_d, total_padded_32_div_32*sizeof(uint32_t),0);
    cudaMallocAsync(&close_count_32_d, total_padded_32_div_32*sizeof(uint32_t),0);

    start = clock();
    assign_open_close<<<numBlock, BLOCKSIZE>>>(block_d, open_d, close_d,  size, total_padded_32);
    cudaStreamSynchronize(0);
    end = clock();
    //time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;

    //printf("assign time: %f\n", time);

    //print8_d<uint8_t>(block_d, size, ROW1);
    //print_d(open_d, total_padded_32, ROW1);
    //print_d(close_d, total_padded_32, ROW1);

    start = clock();
    sum_ones<<<smallNumBlock, BLOCKSIZE>>>(open_d, open_count_d, open_count_32_d, size, total_padded_32, total_padded_32_div_32);
    sum_ones<<<smallNumBlock, BLOCKSIZE>>>(close_d, close_count_d, close_count_32_d, size, total_padded_32, total_padded_32_div_32);
    cudaStreamSynchronize(0);
    end = clock();
    //time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //printf("sum time: %f\n", time);


    //print_d(open_count_d, total_padded_32, ROW1);
    //print_d(close_count_d, total_padded_32, ROW1);

    start = clock();

    thrust::exclusive_scan(thrust::cuda::par, open_count_32_d, open_count_32_d+total_padded_32_div_32, open_count_32_d);
    thrust::exclusive_scan(thrust::cuda::par, close_count_32_d, close_count_32_d+total_padded_32_div_32, close_count_32_d);

    end = clock();
    //time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //printf("prefix time: %f\n", time);

    //print_d(open_count_32_d, total_padded_32_div_32, ROW1);
    //print_d(close_count_32_d, total_padded_32_div_32, ROW1);


    //cudaMallocAsync(&open_prefix_sum_d, total_padded_32*sizeof(uint32_t),0);
    //cudaMallocAsync(&close_prefix_sum_d, total_padded_32*sizeof(uint32_t),0);

    start = clock();

    scatter<<<smallNumBlock, BLOCKSIZE>>>(open_count_32_d, open_count_d, open_prefix_sum_d, total_padded_32_div_32, total_padded_32);
    scatter<<<smallNumBlock, BLOCKSIZE>>>(close_count_32_d, close_count_d, close_prefix_sum_d, total_padded_32_div_32, total_padded_32);
    cudaStreamSynchronize(0);
    end = clock();

    //time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //printf("scatter time: %f\n", time);

    //print_d(open_prefix_sum_d, total_padded_32, ROW1);
    //print_d(close_prefix_sum_d, total_padded_32, ROW1);

    cudaFreeAsync(open_count_d,0);
    cudaFreeAsync(close_count_d,0);
    cudaFreeAsync(open_count_32_d,0);
    cudaFreeAsync(close_count_32_d,0);

    uint32_t* open_block;
    uint32_t* close_block;

    cudaMallocAsync(&open_block, size*sizeof(uint32_t),0);
    cudaMallocAsync(&close_block, size*sizeof(uint32_t),0);

    start = clock();

    scatter_block<<<numBlock, BLOCKSIZE>>>(open_prefix_sum_d, open_d, open_block, total_padded_32, size);
    scatter_block<<<numBlock, BLOCKSIZE>>>(close_prefix_sum_d, close_d, close_block, total_padded_32, size);
    cudaStreamSynchronize(0);

    end = clock();
    //time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //printf("scatter block time: %f\n", time);

    //print_d(open_block, size, ROW1);
    //print_d(close_block, size, ROW1);
    cudaFreeAsync(open_d,0);
    cudaFreeAsync(close_d,0);
    cudaFreeAsync(open_prefix_sum_d,0);
    cudaFreeAsync(close_prefix_sum_d,0);
    
    thrust::transform(thrust::cuda::par, open_block, open_block+size, close_block, open_block, thrust::minus<uint32_t>());

    //print_d(open_block, size, ROW1);

    uint32_t* index_d;

    cudaMallocAsync(&index_d, size*sizeof(uint32_t),0);
    ///TODO Fill from 0 to N-1
    thrust::sequence(thrust::cuda::par, index_d, index_d+size);

    //print_d(index_d, size, ROW1);


    thrust::transform_if(thrust::cuda::par, index_d, index_d+size, open_block, index_d, functor<set_to_zero>(), not_zero());

    //print_d(index_d, size, ROW1);
    
    int sum_zero = thrust::count_if(thrust::cuda::par, index_d, index_d+size, not_zero());
    //printf("%d\n", sum_zero);
    uint32_t* open_close_reduced_d;
    uint32_t* open_close_reduced_out_d;

    cudaMallocAsync(&open_close_reduced_d, sum_zero*sizeof(uint32_t),0);
    cudaFreeAsync(open_block,0);
    cudaFreeAsync(close_block,0);

    thrust::copy_if(thrust::cuda::par, index_d, index_d+size, open_close_reduced_d, not_zero());

    //print_d(open_close_reduced_d, sum_zero, ROW1);
    //open_close_reduced = (uint32_t *) malloc(sizeof(uint32_t)*sum_zero);
    open_close_reduced_size = sum_zero;
    //printf("%d\n", sum_zero);
    open_close_reduced_out_d = open_close_reduced_d;
    //cudaMemcpyAsync(open_close_reduced, open_close_reduced_d, sizeof(uint32_t)*sum_zero, cudaMemcpyDeviceToHost);
    //print(open_close_reduced, sum_zero, ROW1);
    //cudaFreeAsync(open_close_reduced_d,0);
    cudaFreeAsync(index_d,0);
    
    cudaMemcpyAsync(&last_index, open_close_reduced_out_d+sum_zero-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //last_index = open_close_reduced[sum_zero-1];
    if(last_index != size) last_index = last_index+1;
    //printf("%d\n", last_index);
    //last_index = open_close_reduced_d[sum_zero-1];
    //printf("DFSFDSFSD\n");
    printf("*************************************\n");
    return open_close_reduced_out_d;
}


inline uint8_t* multi_to_one_record( uint8_t* tokens_d, uint32_t last_index_tokens){
    clock_t start, end;
    start = clock();

    int numBlock = ((last_index_tokens) + BLOCKSIZE - 1) / BLOCKSIZE;

    uint32_t res_size = last_index_tokens+3;
    //uint8_t* res = (uint8_t*)malloc(sizeof(uint32_t)*(res_size));
    uint8_t* res_d;
    cudaMallocAsync(&res_d, sizeof(uint8_t)*res_size,0);
    //TODO put everything to device array;
    //uint8_t* tokens_d;
    //uint32_t* complete_records_d;
    //cudaMallocAsync(&tokens_d, sizeof(uint8_t)*ret_size,0);
    //cudaMallocAsync(&complete_records_d, sizeof(uint32_t)*open_close_reduced_size_tokens,0);
    //cudaMemcpyAsync(tokens_d, tokens, sizeof(uint8_t)*ret_size, cudaMemcpyHostToDevice);
    //cudaMemcpyAsync(complete_records_d, complete_records, sizeof(uint32_t)*open_close_reduced_size_tokens, cudaMemcpyHostToDevice);

    //print8(tokens, ret_size, ROW1);
    //print_d(complete_records_d, open_close_reduced_size_tokens, ROW1);
    //printf("%d\n", ret_size);
    //printf("%d\n", res_size);

    parallel_copy<<<numBlock, BLOCKSIZE>>>(tokens_d, res_d, res_size, last_index_tokens);
    
    cudaStreamSynchronize(0);

    //cudaMemcpyAsync(res, res_d, res_size*sizeof(uint8_t), cudaMemcpyDeviceToHost);

    end = clock();


    //cudaFreeAsync(tokens_d,0);
    
    //cudaFreeAsync(complete_records_d,0);

    //cudaFreeAsync(res_d,0);

    //printf("inside %f\n", ((double)(end-start)/CLOCKS_PER_SEC)*1000);

    //print8_d<uint8_t>(res_d, res_size, ROW1);
    return res_d;

}


inline void * start(void *start_input){
    uint8_t * block = ((start_input_t *)start_input)->block;
    uint64_t size = ((start_input_t *)start_input)->size;
    uint32_t* res = ((start_input_t *)start_input)->res;
    double total_runtime = ((start_input_t *)start_input)->total_runtime;
    //printf("input size: %ld\n", size);
    //uint8_t * block, uint64_t size, int bLoopCompleted, uint32_t* res, double & total_runtime;
    uint8_t * block_d;
    uint64_t * parse_tree; 
    uint8_t* tokens_d;
    clock_t start, end;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    size_t limit_v;
    float runtime=0, utf_runtime = 0, tokenize_runtime = 0, last_record_runtime = 0, multi_to_one_runtime = 0, parser_runtime=0;
    // cudaDeviceGetLimit(&limit_v, cudaLimitStackSize);
    // printf("Stack Limit: %ld\n", limit_v);
    // cudaDeviceGetLimit(&limit_v, cudaLimitPrintfFifoSize);
    // printf("Print FIFO: %ld\n", limit_v);
    // cudaDeviceGetLimit(&limit_v, cudaLimitMallocHeapSize);
    // printf("Heap Size: %ld\n", limit_v);
    // cudaDeviceGetLimit(&limit_v, cudaLimitDevRuntimeSyncDepth);
    // printf("Runtime Depth: %ld\n", limit_v);
    // cudaDeviceGetLimit(&limit_v, cudaLimitDevRuntimePendingLaunchCount);
    // printf("Runtime Pending Lunch: %ld\n", limit_v);
    // cudaDeviceGetLimit(&limit_v, cudaLimitMaxL2FetchGranularity);
    // printf("Max L2 Fetch: %ld\n", limit_v);
    // cudaDeviceGetLimit(&limit_v, cudaLimitPersistingL2CacheSize);
    // printf("Persisting L2 Cache: %ld\n", limit_v);

    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (1<<10));
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, (1<<10));

    //std::this_thread::sleep_for(std::chrono::seconds(10));
    // cudaDeviceGetLimit(&limit_v, cudaLimitPrintfFifoSize);
    // printf("Print FIFO: %ld\n", limit_v);
    // cudaDeviceGetLimit(&limit_v, cudaLimitMallocHeapSize);
    // printf("Heap Size: %ld\n", limit_v);
 

    //printf("%c\n", (char)block[0]);
    //printf("%c\n", block[38]);
    //start = clock();
    cudaMallocAsync(&block_d, size*sizeof(uint8_t),0);
    cudaMemcpyAsync(block_d, block, sizeof(uint8_t)*size, cudaMemcpyHostToDevice);
    //end = clock();
    //printf("%f\n", ((double)(end-start)/CLOCKS_PER_SEC)*1000);

    start = clock();
    cudaEventRecord(gpu_start);
    bool isValidUTF8 = UTF8Validate(block_d, size);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();
    //utf_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    cudaEventElapsedTime(&utf_runtime, gpu_start, gpu_stop);

    if(!isValidUTF8) {
        printf("not a valid utf input\n"); 
        exit(0);
    }

    //uint32_t* complete_records;
    uint32_t last_index_tokens;
    //uint32_t open_close_reduced_size_tokens;

    start = clock();
    cudaEventRecord(gpu_start);
    int ret_size = 0;
    //TODO Pass everything between blocks. in string, 
    tokens_d = Tokenize(block_d, size, ret_size, last_index_tokens);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();
    //tokenize_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    cudaEventElapsedTime(&tokenize_runtime, gpu_start, gpu_stop);

    uint32_t last_index;
    uint32_t open_close_reduced_size;


    //print8_d<uint8_t>(in_string_8_d, size, ROW1);

    //start = clock();

    //printf("last_index2: %d\n", last_index_tokens);       // is it possible?

    //end = clock();
    //last_record_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;

    uint32_t* parser_input;

    start = clock();
    cudaEventRecord(gpu_start);
    int all_in_one_size = last_index_tokens+3;
    uint8_t* all_in_one_d =  multi_to_one_record(tokens_d, last_index_tokens);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();
    //multi_to_one_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    cudaEventElapsedTime(&multi_to_one_runtime, gpu_start, gpu_stop);

    //int i = last_index;
    //while(tokens[i] != in_string_8[last_index]) i--;
    //i++;
    //printf("i: %d\n", last_index);
    //printf("i: %c\n", in_string_8[i]);
    
    cudaFreeAsync(tokens_d,0);
    cudaFreeAsync(block_d,0);

    //print8_d<uint8_t>(all_in_one_d, all_in_one_size, ROW1);

    start = clock();
    cudaEventRecord(gpu_start);
    NewRuntime_Parallel_GPU((char *)all_in_one_d, all_in_one_size);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();
    parser_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    cudaEventElapsedTime(&parser_runtime, gpu_start, gpu_stop);

                                                        // RUN time ?!
    //print8_d<uint8_t>(all_in_one_d, all_in_one_size, ROW1);
    runtime = utf_runtime + tokenize_runtime + multi_to_one_runtime + parser_runtime;

    cudaStreamSynchronize(0);
    //cudaFreeAsync(complete_records_d,0);
    cudaFreeAsync(all_in_one_d,0);

    /*
    size_t total, free, allocated;
    cudaMemGetInfo(&free, &total);
    allocated = total - free;
    printf("total: %ld, allocated: %ld, free: %ld\n", total, allocated, free);
    */

    printf("utf runtime: %f\n", utf_runtime);
    printf("tokenize runtime: %f\n", tokenize_runtime);
    //printf("last record runtime: %f\n", last_record_runtime);
    printf("multi to one runtime: %f\n", multi_to_one_runtime);
    printf("parser runtime: %f\n", parser_runtime);
    printf("total runtime: %f\n", runtime);
    printf("---------------------------------------------------\n");


    for(double temp = ((start_input_t *)start_input)->total_runtime;
        !(((start_input_t *)start_input)->total_runtime).compare_exchange_strong(temp, temp+runtime););
    //(((start_input_t *)start_input)->total_runtime)+=runtime;

    //return last_index;

    //return 0; //For now
    return NULL;
}

inline uint32_t ** readFilebyLine(char* name){
    clock_t start_time, end_time;
    unsigned long  bytesread;
    //static uint8_t  buf[BUFSIZE];
    static uint8_t*  buf;
    buf = (uint8_t*)malloc(sizeof(uint8_t)*BUFSIZE);
    const int CPU_threads_m_one = CPUTHREADS - 1;
    int   sizeLeftover=0;
    long  pos = 0;
    uint32_t ** res = (uint32_t **)malloc(sizeof(uint32_t *) * CPUTHREADS);
    FILE * handle;
    double total_runtime = 0;

    ssize_t read;
    uint8_t * line = NULL;
    size_t len = 0;
    uint32_t total = 0;
    uint32_t lines = 0;
    uint32_t lineLengths[1<<20];
    int i = 0;


    // Open source file
    if (!(handle = fopen(name,"rb")))
    {
    // Bail
        printf("file not found!\n");
        return 0;
    }

    while((read = getline((char **)&line, &len, handle)) != -1){

        if(total+read > BUFSIZE){
            pthread_t threads[CPUTHREADS];
            start_input_t start_input[CPUTHREADS];
            int sub_input_index = (i+CPU_threads_m_one)/CPUTHREADS;
            start_time = clock();
            for(int j=0; j<CPUTHREADS && j < sub_input_index; j++){
                start_input[j].block = j > 0 ? buf+lineLengths[sub_input_index*j-1] : buf;
                start_input[j].size = j < CPU_threads_m_one ?  (j > 0 ? 
                                            lineLengths[sub_input_index*(j+1)-1] - lineLengths[sub_input_index*j-1] : 
                                            lineLengths[sub_input_index*(j+1)-1] ) :
                                    lineLengths[i-1] - lineLengths[sub_input_index*j-1];
                
                start_input[j].res = res[j];
                if(pthread_create(&threads[j], NULL, start, (void *)& start_input[j])) break;

            }
            for(int j=0; j<CPUTHREADS; j++){
                pthread_join(threads[j], NULL);
            }
            end_time = clock();
            total_runtime += ((double)(end_time-start_time)/CLOCKS_PER_SEC)*1000;
            //start(buf, total, res, total_runtime);
            total = 0;
            i = 0;
            memcpy(buf+total, line, sizeof(uint8_t)*read);
            total = read; //Reset
            lineLengths[i] = total;
        }
        else{
            memcpy(buf+total, line, sizeof(uint8_t)*read);
            total += read;
            lineLengths[i] = total;

        }
        i++;

    }
    if(total > 0){
        //print8(buf, total, ROW1);
        pthread_t threads[CPUTHREADS];
        start_input_t start_input[CPUTHREADS];
        int sub_input_index = (i+CPU_threads_m_one)/CPUTHREADS;
        start_time = clock();
        for(int j=0; j<CPUTHREADS && j < sub_input_index; j++){
            start_input[j].block = j > 0 ? buf+lineLengths[sub_input_index*j-1] : buf;
            start_input[j].size = j < CPU_threads_m_one ?  (j > 0 ? 
                                        lineLengths[sub_input_index*(j+1)-1] - lineLengths[sub_input_index*j-1] : 
                                        lineLengths[sub_input_index*(j+1)-1] ) :
                                lineLengths[i-1] - lineLengths[sub_input_index*j-1];
            
            start_input[j].res = res[j];
            if(pthread_create(&threads[j], NULL, start, (void *)& start_input[j])) break;

        }
        for(int j=0; j<CPUTHREADS && j < sub_input_index; j++){
            pthread_join(threads[j], NULL);
        }
        end_time = clock();
        total_runtime += ((double)(end_time-start_time)/CLOCKS_PER_SEC)*1000;
        //start(buf, total, res, total_runtime);
    }
    total = 0;
    
    printf("==================== total runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", total_runtime); //start_input_t::total_runtime.load()
    printf("==================== total runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::total_runtime.load()/CPUTHREADS); //start_input_t::total_runtime.load()

    fclose(handle);
    return res;
 
}



inline uint32_t *readFile(char* name){
    clock_t start_time, end_time;
    unsigned long  bytesread;
    static uint8_t  buf[BUFSIZE];
    int   sizeLeftover=0;
    int   bLoopCompleted = 0;
    long  pos = 0;
    uint32_t * res;
    FILE * handle;
    double total_runtime = 0;
    
    // Open source file
    if (!(handle = fopen(name,"rb")))
    {
    // Bail
        printf("file not found!\n");
        return 0;
    }

    do
    {
   
    // Read next block from file and save into buf, right after the
    // "left over" buffer
        bytesread = fread(buf+sizeLeftover, 1, sizeof(buf)-1-sizeLeftover, handle);
        if (bytesread<1)
        {
            // Turn on 'loop completed' flag so that we know to exit at the bottom
            // Still need to process any block we currently have in the
            // leftover buffer
            bLoopCompleted = 1;
            bytesread  = 0;
        }     
   
    // Add NULL terminator at the end of our buffer
        buf[bytesread+sizeLeftover] = 0;   
        //sizeLeftover>0 ?  print8(buf, sizeLeftover, ROW1) : NULL;

    // Process data - Replace with your function
    //
    // Function should return the position in the file or -1 if failed
    //
    // We are also passing bLoopCompleted to let ProcessData know whether this is
    // the last record (in which case - if no end-of-record separator,
    // use eof and process anyway)
        //start_time = clock();
        pos = 0; //start(buf, bytesread+sizeLeftover, bLoopCompleted, res, total_runtime);
        //end_time = clock();
        //total_runtime +=  ((double)(end_time-start_time)/CLOCKS_PER_SEC)*1000;


    // If error occured, bail
        if (pos<1) 
        {
            bLoopCompleted = 1;
            pos      = 0;
        }
   
    // Set Left over buffer size to
    //
    //  * The remaining unprocessed buffer that was not processed
    //  by ProcessData (because it couldn't find end-of-line)
    //
    // For protection if the remaining unprocessed buffer is too big
    // to leave sufficient room for a new line (MAXLINELENGTH), cap it
    // at maximumsize - MAXLINELENGTH
        //printf("%ld\n", bytesread);
        //printf("%d\n", sizeLeftover);
        //printf("%ld\n", pos);

        //printf("%ld\n", bytesread+sizeLeftover-pos);
        //printf("%ld\n", sizeof(buf)-MAXLINELENGTH);
        sizeLeftover = bytesread+sizeLeftover-pos;
        //printf("%d\n", sizeLeftover);
    // Extra protection - should never happen but you can never be too safe
        if (sizeLeftover<1) sizeLeftover=0;

    // If we have a leftover unprocessed buffer, move it to the beginning of 
    // read buffer so that when reading the next block, it will connect to the
    // current leftover and together complete a full readable line
        if (pos!=0 && sizeLeftover!=0)
        memmove(buf, buf+pos, sizeLeftover);
        
        
    } while(!bLoopCompleted);
    // Close file
    
    printf("==================== total runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", total_runtime);

    fclose(handle);
    return res;
 
}


int main(int argc, char **argv)
{
    uint32_t** result;
  if (argv[1] != NULL){
    if( strcmp(argv[1], "-b") == 0 && argv[2] != NULL){
      std::cout << "Batch mode..." << std::endl;
      //result = readFile(argv[2]);
      result = readFilebyLine(argv[2]);
    }
    else std::cout << "Command should be like '-b[file path]'" << std::endl;
  }
  else{
    std::cout << "Please select (batch: -b): " << std::endl;
  }
  cudaDeviceReset();
  return 0;
}