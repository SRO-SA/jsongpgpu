#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
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
#include "JSON_Parallel_Parser.h"
#include <thrust/transform.h>
#include <inttypes.h>

#define        MAXLINELENGTH    67108864  //4194304  //8388608 33554432 67108864 134217728 536870912 1073741824// Max record size
#define        BUFSIZE       67108864  //4194304 //8388608 33554432 67108864 134217728 536870912 1073741824

enum hypothesis_val {in, out, unknown, fail};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define RUNTIMES 1

#define BLOCKSIZE 256
#define FILESCOUNT 4
#define NAMELENGTH 25

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

struct not_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x > 0);
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

int print(uint32_t* input, int length, int rows){
    for(long i =0; i<rows; i++){
      for(long j=0; j<length; j++){
        std::cout << *(input+j+(i*length)) << ' ';
      }
      std::cout << std::endl;
    }
    return 1;
}

int print_d(uint32_t* input_d, int length, int rows){
    uint32_t * input;
    input = (uint32_t*) malloc(sizeof(uint32_t)*length*rows);
    cudaMemcpy(input, input_d, sizeof(uint32_t)*length*rows, cudaMemcpyDeviceToHost);

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
    cudaMemcpy(input, input_d, sizeof(uint8_t)*length, cudaMemcpyDeviceToHost);

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

__global__
void permute_32(uint8_t *block_d, uint32_t *permutation_output_d, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int prev = (i-1)*4;
        int start = i*4;
        
        permutation_output_d[i] = i!=0 ?  block_d[prev+2] | (block_d[prev+3] << 8) | (block_d[start] << 16) | (block_d[start+1] << 24) :
                                        0 | 0 << 8 | (block_d[start] << 16) | (block_d[start+1] << 24);
    }
}

__global__
void align_right(uint32_t* permutation_output_d, uint8_t* block_d, uint32_t* prev_d, uint64_t size, int total_padded_32, int shift){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int prev = (i-1)*4;
        int start = i*4;
        uint64_t dist = i > 0 ? ((uint64_t)((block_d[start+3] << 24) | (block_d[start+2] << 16) | (block_d[start+1] << 8) | (block_d[start])) << 32) |
                        ((block_d[prev+3] << 24) | (block_d[prev+2] << 16) | (block_d[prev+1] << 8) | (block_d[prev]))
                        : ((uint64_t)((block_d[start+3] << 24) | (block_d[start+2] << 16) | (block_d[start+1] << 8) | (block_d[start])) << 32) |
                        (0);

        //if(i==8800) printf("align: %ld\n", dist);
        //uint32_t dist1 = (((uint32_t)block_d[start] << 16 | (uint32_t)block_d[start+1] << 24)
        //| (permutation_output_d[i] & 0x0000ffff)) >> shift*8; // ((uint32_t)block_d[start] << 24 | (uint32_t)block_d[start+1] << 16) original
        //uint32_t dist2 = (((uint32_t)block_d[start+2] << 16 | (uint32_t)block_d[start+3] << 24) 
        //| ((permutation_output_d[i] & 0xffff0000) >> 16)) >> shift*8; // ((uint32_t)block_d[start+2] << 24 | (uint32_t)block_d[start+3] << 16) original

        
    
        prev_d[i] = (uint32_t)(dist >> shift*8);
    }

}


__global__ 
void is_incomplete(uint8_t* block, uint32_t* prev_incomplete_d, uint64_t size, int total_padded_32){
    static const uint32_t max_val = (uint32_t)(0b11000000u-1 << 24) | (uint32_t)(0b11100000u-1 << 16) | (uint32_t)(0b11110000u-1 << 8) | (uint32_t)(255);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //int prev = (i-1)*4;
        //printf("prev: %ld\n", i);
        int start = i*4;
        uint32_t val = (uint32_t)block[start] | (uint32_t)(block[start+1] << 8) | (uint32_t)(block[start+2] << 16) | (uint32_t)(block[start+3] << 24);
        prev_incomplete_d[i] = __vsubus4(val, max_val);
        //printf("now %ld\n", i);
    }
}

__global__
void check_incomplete_ascii(uint8_t* block, uint32_t* prev_incomplete_d, uint32_t* is_ascii_d, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //int prev = (i-1)*4;
        int start = i*4;
        is_ascii_d[i] = (uint32_t)(((block[start]&0b10000000) >> 7)
                    | ((block[start+1]&0b10000000) >> 6)
                    | ((block[start+2]&0b10000000) >> 5)
                    | ((block[start+3]&0b10000000) >> 4)) == 0;
        //if(!is_ascii_d[i]) printf("index: %d, is_ascii_d: %d\n", i, is_ascii_d[i]);

        if(is_ascii_d[i] && i!=0) {is_ascii_d[i] = prev_incomplete_d[i-1];}
        else {is_ascii_d[i] = 0;}
    }

}

__global__
void must_be_2_3_continuation(uint32_t* prev1, uint32_t* prev2, uint32_t* must32_d, uint64_t size, int total_padded_32){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //int prev = (i-1)*4;
        //int start = i*4;
        uint32_t third_subtract_byte = (0b11100000u-1) | (0b11100000u-1) << 8 | (0b11100000u-1) << 16 | (0b11100000u-1) << 24;
        uint32_t fourth_subtract_byte = (0b11110000u-1) | (0b11110000u-1) << 8 | (0b11110000u-1) << 16 | (0b11110000u-1) << 24;
        uint32_t is_third_byte = __vsubus4(prev1[i], third_subtract_byte);
        uint32_t is_fourth_byte = __vsubus4(prev2[i], fourth_subtract_byte);
        /*if(i==8880){
            printf("Third %d\n", is_third_byte);
            printf("Fourth %d\n", is_fourth_byte);
        }*/

        uint32_t gt = __vsubss4((int32_t)(is_third_byte | is_fourth_byte), int32_t(0));
        uint8_t gt1 = (int8_t)gt & 0xFF > 0 ? 0xFF : 0;
        uint8_t gt2 =  (int8_t)(gt >> 8) & 0xFF > 0 ? 0xFF : 0;
        uint8_t gt3 =  (int8_t)(gt >> 16) & 0xFF > 0 ? 0xFF : 0;
        uint8_t gt4 =  (int8_t)(gt >> 24) & 0xFF > 0 ? 0xFF : 0;

        must32_d[i] = gt1 | (gt2 << 8) | (gt3 << 16) | (gt4 << 24);
        /*if(i==0){
            printf("gt: %u\n", gt);
        }*/
    }
}

__global__
void check_special_cases(uint8_t* block_d, uint32_t* prev1_d, uint32_t* sc_d, uint64_t size, int total_padded_32){
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
    uint8_t table1[16] = {      
        // 0_______ ________ <ASCII in byte 1>
        TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
        TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
        // 10______ ________ <continuation in byte 1>
        TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
        // 1100____ ________ <two byte lead in byte 1>
        TOO_SHORT | OVERLONG_2,
        // 1101____ ________ <two byte lead in byte 1>
        TOO_SHORT,
        // 1110____ ________ <three byte lead in byte 1>
        TOO_SHORT | OVERLONG_3 | SURROGATE,
        // 1111____ ________ <four+ byte lead in byte 1>
        TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4
    };
    constexpr const uint8_t CARRY = TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 . 10000011
    uint8_t table2[16] = {
          // ____0000 ________
          CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4, //11100111
          // ____0001 ________
          CARRY | OVERLONG_2,
          // ____001_ ________
          CARRY,
          CARRY,
    
          // ____0100 ________
          CARRY | TOO_LARGE,
          // ____0101 ________
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          // ____011_ ________
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          CARRY | TOO_LARGE | TOO_LARGE_1000,
    
          // ____1___ ________
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          // ____1101 ________
          CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
          CARRY | TOO_LARGE | TOO_LARGE_1000,
          CARRY | TOO_LARGE | TOO_LARGE_1000
    };
    uint8_t table3[16] = {
      // ________ 0_______ <ASCII in byte 2>
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,

      // ________ 1000____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
      // ________ 1001____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
      // ________ 101_____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE  | TOO_LARGE,
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE  | TOO_LARGE,

      // ________ 11______
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT
    };
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //int prev = (i-1)*4;
        int start = i*4;
        uint32_t shr_prev1 = (prev1_d[i] >> 4) & 0x0f0f0f0f;
        //if(i==8880) printf("prev1_d: %u\n", prev1_d[i]);

        //if(i==8880) printf("shr_prev1: %u\n", shr_prev1);

        uint8_t shr_prev1_b1 = shr_prev1 & 0x000000ff;
        uint8_t shr_prev1_b2 = (shr_prev1 >> 8) & 0x000000ff;
        uint8_t shr_prev1_b3 = (shr_prev1 >> 16) & 0x000000ff;
        uint8_t shr_prev1_b4 = (shr_prev1 >> 24) & 0x000000ff;

        uint32_t byte_1_high = table1[shr_prev1_b1] | (table1[shr_prev1_b2] << 8) | (table1[shr_prev1_b3] << 16) | (table1[shr_prev1_b4] << 24);

        uint32_t shl_prev1 = prev1_d[i] & 0x0f0f0f0f;
         shr_prev1_b1 = shl_prev1;
         shr_prev1_b2 = (shl_prev1 >> 8);
         shr_prev1_b3 = (shl_prev1 >> 16);
         shr_prev1_b4 = (shl_prev1 >> 24);

        uint32_t byte_1_low = table2[shr_prev1_b1] | (table2[shr_prev1_b2] << 8) | (table2[shr_prev1_b3] << 16) | (table2[shr_prev1_b4] << 24);
        //if(i==8880) printf("shl_prev1: %u\n", shl_prev1);

        shr_prev1_b1 = (block_d[start] >> 4);
        shr_prev1_b2 = (block_d[start+1] >> 4);
        shr_prev1_b3 = (block_d[start+2] >> 4);
        shr_prev1_b4 = (block_d[start+3] >> 4);

        uint32_t byte_2_high = table3[shr_prev1_b1] | (table3[shr_prev1_b2] << 8) | (table3[shr_prev1_b3] << 16) | (table3[shr_prev1_b4] << 24);
        sc_d[i] =   (byte_1_high & byte_1_low & byte_2_high);
        //if(i==8880) printf("8880: %d, byte_1_high: %u, byte_1_low: %u, byte_2_high: %u\n", sc_d[i], byte_1_high, byte_1_low, byte_2_high);
        //if(i==8880) printf("prev[start+3]: %d, block[start]: %u, block[start+1]: %c, block[start+2]: %c, block[start+3]: %c\n",
        //             block_d[(i-1)*4+3], block_d[start], block_d[start+1], block_d[start+2], block_d[start+3]);
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

}



uint8_t prefix_or(uint32_t* is_ascii_d, uint64_t size, int total_padded_32){

    thrust::inclusive_scan(thrust::cuda::par, is_ascii_d, is_ascii_d + total_padded_32, is_ascii_d);
    uint32_t error;
    cudaMemcpy(&error, is_ascii_d+total_padded_32-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("error %d\n", error);

    return (uint8_t)error;
}
  

bool UTF8Validate(uint8_t * block_d, uint64_t size){
    int total_padded_32 = ((size + 3)/4) ;
    uint32_t* is_ascii_d;
    uint32_t* permutation_output_d;
    uint32_t* prev_incomplete_d;
    int numBlock = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;

    //print8_d<uint8_t>(block_d, (int)size, ROW1);

    cudaMalloc(&is_ascii_d, sizeof(uint32_t)*total_padded_32);
    cudaMalloc(&prev_incomplete_d, sizeof(uint32_t)*total_padded_32);

    is_incomplete<<<numBlock, BLOCKSIZE>>>(block_d, prev_incomplete_d, size, total_padded_32);
    cudaDeviceSynchronize();

    //print_d(prev_incomplete_d, total_padded_32, ROW1);

    check_incomplete_ascii<<<numBlock, BLOCKSIZE>>>(block_d, prev_incomplete_d, is_ascii_d, size, total_padded_32);
    cudaDeviceSynchronize();




    uint8_t error = prefix_or(is_ascii_d, size, total_padded_32);

    //printf("error %d\n", error);

    //std::cout << "HERE" << std::endl;

    if(error != 0){ printf("Incomplete ASCII!\n"); return false;}

    cudaMalloc(&permutation_output_d, sizeof(uint32_t)*total_padded_32);
    permute_32<<<numBlock, BLOCKSIZE>>>(block_d, permutation_output_d, size, total_padded_32);
    cudaDeviceSynchronize();

    //print_d(permutation_output_d, total_padded_32, ROW1);

    uint32_t* prev1_d=is_ascii_d, *sc_d=prev_incomplete_d;
    //cudaMalloc(&prev1, sizeof(uint32_t)*total_padded_32);
    //cudaMalloc(&sc, sizeof(uint32_t)*total_padded_32);
    align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev1_d, size, total_padded_32, 4-1);
    cudaDeviceSynchronize();
    //printf("align1\n");
    //print_d(prev1_d, total_padded_32, ROW1);
    check_special_cases<<<numBlock, BLOCKSIZE>>>(block_d, prev1_d, sc_d, size, total_padded_32);
    cudaDeviceSynchronize();
    //printf("special\n");
    //print_d(sc_d, total_padded_32, ROW1);

    cudaFree(prev1_d);
    uint32_t* prev2_d = prev1_d, *prev3_d;
    cudaMalloc(&prev2_d, sizeof(uint32_t)*total_padded_32);
    cudaMalloc(&prev3_d, sizeof(uint32_t)*total_padded_32);
    align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev2_d, size, total_padded_32, 4-2);
    cudaDeviceSynchronize();
    //printf("align2\n");
    //print_d(prev2_d, total_padded_32, ROW1);

    align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev3_d, size, total_padded_32, 4-3);
    cudaDeviceSynchronize();
    //printf("align3\n");
    //print_d(prev3_d, total_padded_32, ROW1);
    cudaFree(permutation_output_d);

    uint32_t* must32_d;
    cudaMalloc(&must32_d, sizeof(uint32_t)*total_padded_32);
    must_be_2_3_continuation<<<numBlock, BLOCKSIZE>>>(prev2_d, prev3_d, must32_d, size, total_padded_32);
    cudaDeviceSynchronize();
    //printf("must32_d\n");
    //print_d(must32_d, total_padded_32, ROW1);
    cudaFree(prev2_d);
    cudaFree(prev3_d);

    uint32_t* must32_80_d, *must32_80_sc_d;
    cudaMalloc(&must32_80_d, sizeof(uint32_t)*total_padded_32);
    cudaMalloc(&must32_80_sc_d, sizeof(uint32_t)*total_padded_32);

    single_parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(must32_d, 0x80, must32_80_d, size, total_padded_32);
    cudaDeviceSynchronize();
    //printf("must32_80_d\n");
    //print_d(must32_80_d, total_padded_32, ROW1);
    cudaFree(must32_d);
    uint32_t test_value;
    cudaMemcpy(&test_value, must32_80_d+8880, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("must32_80: %u\n", test_value);
    cudaMemcpy(&test_value, sc_d+8880, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("sc_d: %d\n", test_value);

    parallel_xor<<<numBlock, BLOCKSIZE>>>(must32_80_d, sc_d, must32_80_sc_d, size, total_padded_32);
    cudaDeviceSynchronize();

    cudaMemcpy(&test_value, must32_80_sc_d+8880, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("must32_80_sc_d: %u\n", test_value);


    cudaFree(must32_80_d);
    cudaFree(sc_d);

    error = prefix_or(must32_80_sc_d, size, total_padded_32);
    cudaFree(must32_80_sc_d);
    return !(error);

}

__global__
void get(uint8_t* block_d, uint32_t* output, uint64_t size, int total_padded_32, int what){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int start = i*32;
        uint32_t res = 0;

        for (int j = start; j<start+32; j++){
            switch(what){
                case 0:
                    if(block_d[j] == '\\') res |= 1 << (j-start);
                break;
                case 1:
                    if(block_d[j] == '\"') res |= 1 << (j-start);
                break;
                case 2:
                break;
            }
        }
        output[i] = res;
    }
}

__global__
void split(uint32_t* input, uint8_t* output, uint64_t size, int total_padded_32){
    uint8_t val_arr[2] = {0x00, 0xff};
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        uint32_t val = input[i];
        int start = i*32;
        for(int j=0; j<32 && j+start<size; j++){
            output[start+j] = val_arr[((val >> (j)) & 0b00000001)];
        }
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int start = i*32;
        uint32_t res_op = 0;
        uint32_t res_wt = 0;
        for (int j = start; j<start+32; j++){
            res_op |= (((block_d[j] == '{' ||
                    block_d[j] == '[' ||
                    block_d[j] == '}' ||
                    block_d[j] == ']' ||
                    block_d[j] == ':' ||
                    block_d[j] == ','
                    ) ? 1 : 0) << (j-start)) ;
            res_wt |= (((block_d[j] == ' ' ||
                    block_d[j] == '\t' ||
                    //block_d[j] == '\n' ||
                    block_d[j] == '\r'
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
            uint32_t follows_escape_t = backslashes_d[j] << 1;
            uint32_t odd_seq_t = backslashes_d[j] & ~even_bits & ~follows_escape_t;
            uint32_t last_zero = ~(backslashes_d[j] | odd_seq_t);
            uint32_t last_one = backslashes_d[j] & odd_seq_t;
            //if(i==882) printf("%u\n", backslashes_d[j]);

            //if(i==882) printf("%u\n", last_zero);
            //if(i==882) printf("%u\n", last_one);

            uint32_t last_two_bits = (backslashes_d[j] & 0xC0000000UL) >> 30;
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

uint8_t * Tokenize(uint8_t* block_d, uint64_t size, int &ret_size, uint8_t*& in_string_8_out_d, uint32_t &last_index_tokens){
    int total_padded_32 = (size+31)/32 ;
    int numBlock = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    uint32_t* backslashes_d;
    //print8_d<uint8_t>(block_d, size, ROW1);
    clock_t start, end;
    start = clock();
    cudaMalloc(&backslashes_d, total_padded_32 * sizeof(uint32_t));
    get<<<numBlock, BLOCKSIZE>>>(block_d, backslashes_d, size, total_padded_32, 0);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(backslashes_d, total_padded_32, ROW1);
    //printf("%d\n", total_padded_32);

    uint32_t* escaped_d;
    start = clock();
    cudaMalloc(&escaped_d, total_padded_32 * sizeof(uint32_t));
    find_escaped<<<numBlock, BLOCKSIZE>>>(backslashes_d, escaped_d, total_padded_32);
    cudaDeviceSynchronize();
    cudaFree(backslashes_d);
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //printf("ecaped backslashes: \n");
    //print_d(escaped_d, total_padded_32, ROW1);

    uint32_t* quote_d;
    start = clock();
    cudaMalloc(&quote_d, total_padded_32 * sizeof(uint32_t));
    get<<<numBlock, BLOCKSIZE>>>(block_d, quote_d, size, total_padded_32, 1);//////////////
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(quote_d, total_padded_32, ROW1);

    uint32_t* real_quote_d;
    start = clock();
    cudaMalloc(&real_quote_d, total_padded_32 * sizeof(uint32_t));
    parallel_not<<<numBlock, BLOCKSIZE>>>(escaped_d, escaped_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(escaped_d, total_padded_32, ROW1);
    start = clock();
    parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(quote_d, escaped_d, real_quote_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    cudaFree(quote_d);
    cudaFree(escaped_d);
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(real_quote_d, total_padded_32, ROW1);
    /*
    start = clock();
    uint32_t* prediction_d;
    cudaMalloc(&prediction_d, total_padded_32 * sizeof(uint32_t));
    predict<<<numBlock, BLOCKSIZE>>>(block_d, real_quote_d, prediction_d, size, total_padded_32);
    cudaDeviceSynchronize();
    cudaFree(real_quote_d);
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    */

    //print_d(prediction_d, total_padded_32, ROW1);

    
    uint32_t* in_string_d;
    cudaMalloc(&in_string_d, total_padded_32 * sizeof(uint32_t));
    
    uint32_t* total_one_d;
    uint32_t* total_one_32_d;
    start = clock();
    int total_padded_32_div_32 = (total_padded_32+31)/32 ;

    cudaMalloc(&total_one_d, total_padded_32*sizeof(uint32_t));
    cudaMalloc(&total_one_32_d, (total_padded_32_div_32)*sizeof(uint32_t));

    int smallNumBlock = (total_padded_32_div_32 + BLOCKSIZE - 1) / BLOCKSIZE;

    sum_ones<<<smallNumBlock, BLOCKSIZE>>>(real_quote_d, total_one_d, total_one_32_d,  size, total_padded_32, total_padded_32_div_32);
    cudaDeviceSynchronize();
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
    cudaMalloc(&prefix_sum_ones, total_padded_32*sizeof(uint32_t));
    start = clock();

    scatter<<<smallNumBlock, BLOCKSIZE>>>(total_one_32_d, total_one_d, prefix_sum_ones, total_padded_32_div_32, total_padded_32);
    cudaDeviceSynchronize();

    end = clock();
    //print_d(total_one_d, total_padded_32, ROW1);
    //std::cout << "Time elapsed scatter: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;



    start = clock();
    fact<<<numBlock, BLOCKSIZE>>>(real_quote_d, prefix_sum_ones, in_string_d, total_padded_32_div_32, total_padded_32);
    cudaDeviceSynchronize();
    cudaFree(total_one_d);
    cudaFree(total_one_32_d);
    cudaFree(prefix_sum_ones);
    cudaFree(real_quote_d);
    end = clock();

    //print_d(in_string_d, total_padded_32, ROW1);

    //std::cout << "Time elapsed Fact: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    /*
    cudaMemcpy(in_string_d, prediction_d, sizeof(uint32_t)*total_padded_32, cudaMemcpyDeviceToDevice);
    cudaFree(prediction_d);
    */
    /////////*
    /*
    uint8_t* real_quote_8_d;
    cudaMalloc(&real_quote_8_d, size * sizeof(uint8_t));
    split<<<numBlock, BLOCKSIZE>>>(real_quote_d, real_quote_8_d, size, total_padded_32);
    cudaDeviceSynchronize();
    thrust::inclusive_scan(thrust::cuda::par, real_quote_8_d, real_quote_8_d + size, real_quote_8_d);
    single_parallel_and<uint8_t><<<numBlock, BLOCKSIZE>>>(real_quote_8_d, 0b00000001, real_quote_8_d, size, total_padded_32);
    cudaDeviceSynchronize();
    uint32_t* in_string_d;
    cudaMalloc(&in_string_d, total_padded_32 * sizeof(uint32_t));
    compact<<<numBlock, BLOCKSIZE>>>(real_quote_8_d, in_string_d, size, total_padded_32);
    cudaDeviceSynchronize();
    
    //////////

    ///////////////////////////////////////
    

    prefix_xor(real_quote_d, in_string_t, size, total_padded_32);
    uint32_t* last_bits_d;
    cudaMalloc(&last_bits_d, total_padded_32 * sizeof(uint32_t));
    get_last_bits<<<>>>(in_string_t, last_bits_d, total_padded_32);///////////////
    uint32_t* last_bit_propagate_d;
    cudaMalloc(&last_bit_propagate_d, total_padded_32 * sizeof(uint32_t));
    prefix_xor(last_bits_d, last_bit_propagate_d, size, total_padded_32);
    uint32_t* in_string_d;
    cudaMalloc(&in_string_d, total_padded_32*sizeof(uint32_t));
    prefix_xor(real_quote_d, in_string_d, size, total_padded_32);
    parallel_xor<<<>>>(in_string_d, last_bit_propagate_d, in_string_d, size, total_padded_32);
    */
    /////////////////////////////////////////////
    
    uint32_t *whitespace_d, *op_d;
    cudaMalloc(&whitespace_d, total_padded_32*sizeof(uint32_t));
    cudaMalloc(&op_d, total_padded_32*sizeof(uint32_t));
    start = clock();
    classify<<<numBlock, BLOCKSIZE>>>(block_d, op_d, whitespace_d, size, total_padded_32);////////////////////
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    //print_d(whitespace_d, total_padded_32, ROW1);
    //print_d(op_d, total_padded_32, ROW1);
    
    uint32_t* scalar_d;
    start = clock();
    cudaMalloc(&scalar_d, total_padded_32*sizeof(uint32_t));
    parallel_or<<<numBlock, BLOCKSIZE>>>(op_d, whitespace_d, scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    
    //print_d(scalar_d, total_padded_32, ROW1);
    start = clock();
    parallel_not<<<numBlock, BLOCKSIZE>>>(scalar_d, scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    
    //print_d(scalar_d, total_padded_32, ROW1);

    uint32_t* nonquote_scalar_d;
    cudaMalloc(&nonquote_scalar_d, total_padded_32*sizeof(uint32_t));
    start = clock();
    parallel_not<<<numBlock, BLOCKSIZE>>>(in_string_d, in_string_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    //print_d(in_string_d, total_padded_32, ROW1);
    
    start = clock();
    parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(scalar_d, in_string_d, nonquote_scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    

    //print_d(nonquote_scalar_d, total_padded_32, ROW1);
    
    uint32_t* overflow;
    cudaMalloc(&overflow, total_padded_32*sizeof(uint32_t));
    start = clock();
    parallel_shift_right<<<numBlock, BLOCKSIZE>>>(nonquote_scalar_d, overflow, 31, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    //print_d(overflow, total_padded_32, ROW1);
    
    uint32_t* follows_nonquote_scalar_d;
    cudaMalloc(&follows_nonquote_scalar_d, total_padded_32*sizeof(uint32_t));
    start = clock();
    parallel_shift_left<<<numBlock, BLOCKSIZE>>>(nonquote_scalar_d, follows_nonquote_scalar_d,  1, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(follows_nonquote_scalar_d, total_padded_32, ROW1);

    start = clock();
    parallel_or<<<numBlock, BLOCKSIZE>>>(follows_nonquote_scalar_d, overflow, follows_nonquote_scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(follows_nonquote_scalar_d, total_padded_32, ROW1);

    cudaFree(op_d);
    cudaFree(follows_nonquote_scalar_d);                                                    //Remove later
    cudaFree(scalar_d);
    cudaFree(nonquote_scalar_d);
    cudaFree(overflow);

    start = clock();
    parallel_not<<<numBlock, BLOCKSIZE>>>(whitespace_d, whitespace_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    
    //print_d(whitespace_d, total_padded_32, ROW1);


    start = clock();
    parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(in_string_d, whitespace_d, in_string_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    cudaFree(whitespace_d);                                                         //// let it be here for now
    //print_d(in_string_d, total_padded_32, ROW1);


    uint8_t* in_string_8_d;
    cudaMalloc(&in_string_8_d, size * sizeof(uint8_t));
    start = clock();
    split<<<numBlock, BLOCKSIZE>>>(in_string_d, in_string_8_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    cudaFree(in_string_d);
    //print_d(in_string_d, total_padded_32, ROW1);
    //print8_d<int>(in_string_8_d, size, ROW1);




    uint8_t* in_string_out_d;
    start = clock();
    parallel_and<uint8_t><<<numBlock, BLOCKSIZE>>>(in_string_8_d, block_d, in_string_8_d, size, size);
    cudaDeviceSynchronize();
    end = clock();
    //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //in_string_8  = (uint8_t*) malloc(size*sizeof(uint8_t));
    //cudaMemcpy(in_string_8, in_string_8_d, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost);
    in_string_8_out_d = in_string_8_d;

    uint8_t* filtered_string_8_d;
    int sum = thrust::count_if(thrust::cuda::par, in_string_8_d, in_string_8_d+size, not_zero());
    cudaMalloc(&filtered_string_8_d, sum*sizeof(uint8_t));

    thrust::copy_if(thrust::cuda::par, in_string_8_d, in_string_8_d+size, filtered_string_8_d, not_zero());

    uint8_t* in_string_out;
    in_string_out = (uint8_t* )malloc(sum*sizeof(uint8_t));
    //in_string_out[0] = 'A';

    //cudaMemcpy(in_string_out, filtered_string_8_d, sizeof(uint8_t), cudaMemcpyDeviceToHost);

    //printf("filtered: %c\n", in_string_out[0]);

    //print8(in_string_8, size, ROW1);

    cudaMemcpy(in_string_out, filtered_string_8_d, sizeof(uint8_t)*sum, cudaMemcpyDeviceToHost);

    int total_line = 0;
    int line_length = 0;
    for(int i=0; i<sum; i++){
        if(in_string_out[i] == '\n'){
            line_length++;
            total_line++;
        }
        else line_length++;
    }
    last_index_tokens = line_length;
    //printf("last_index3: %d, open_close_size3: %d\n", line_length, total_line);
    free(in_string_out);
    in_string_out_d = filtered_string_8_d;
    ret_size = sum;

    return in_string_out_d;
}


uint32_t* get_last_record(uint8_t* block_d, int size, uint32_t &last_index, uint32_t& open_close_reduced_size){
    clock_t start, end;
    double time = 0;
    
    uint32_t* open_d;
    uint32_t* close_d;

    uint32_t* open_count_d;
    uint32_t* close_count_d;

    uint32_t* open_count_32_d;
    uint32_t* close_count_32_d;

    uint32_t* open_prefix_sum_d;
    uint32_t* close_prefix_sum_d;

    //uint8_t* block_d;
    int total_padded_32 = (size+31)/32 ;
    int numBlock = (size + BLOCKSIZE - 1) / BLOCKSIZE;
    int total_padded_32_div_32 = (total_padded_32+31)/32 ;
    int smallNumBlock = (total_padded_32_div_32 + BLOCKSIZE - 1) / BLOCKSIZE;


    //cudaMalloc(&block_d, size*sizeof(uint8_t));
    //cudaMemcpy(block_d, block, size*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMalloc(&open_d, total_padded_32*sizeof(uint32_t));
    cudaMalloc(&close_d, total_padded_32*sizeof(uint32_t));

    cudaMalloc(&open_count_d, total_padded_32*sizeof(uint32_t));
    cudaMalloc(&close_count_d, total_padded_32*sizeof(uint32_t));

    cudaMalloc(&open_count_32_d, total_padded_32_div_32*sizeof(uint32_t));
    cudaMalloc(&close_count_32_d, total_padded_32_div_32*sizeof(uint32_t));

    start = clock();
    assign_open_close<<<numBlock, BLOCKSIZE>>>(block_d, open_d, close_d,  size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;

    printf("assign time: %f\n", time);

    //print8_d<uint8_t>(block_d, size, ROW1);
    //print_d(open_d, total_padded_32, ROW1);
    //print_d(close_d, total_padded_32, ROW1);

    start = clock();
    sum_ones<<<smallNumBlock, BLOCKSIZE>>>(open_d, open_count_d, open_count_32_d, size, total_padded_32, total_padded_32_div_32);
    sum_ones<<<smallNumBlock, BLOCKSIZE>>>(close_d, close_count_d, close_count_32_d, size, total_padded_32, total_padded_32_div_32);
    cudaDeviceSynchronize();
    end = clock();
    time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    printf("sum time: %f\n", time);


    //print_d(open_count_d, total_padded_32, ROW1);
    //print_d(close_count_d, total_padded_32, ROW1);

    start = clock();

    thrust::exclusive_scan(thrust::cuda::par, open_count_32_d, open_count_32_d+total_padded_32_div_32, open_count_32_d);
    thrust::exclusive_scan(thrust::cuda::par, close_count_32_d, close_count_32_d+total_padded_32_div_32, close_count_32_d);

    end = clock();
    time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    printf("prefix time: %f\n", time);

    //print_d(open_count_32_d, total_padded_32_div_32, ROW1);
    //print_d(close_count_32_d, total_padded_32_div_32, ROW1);


    cudaMalloc(&open_prefix_sum_d, total_padded_32*sizeof(uint32_t));
    cudaMalloc(&close_prefix_sum_d, total_padded_32*sizeof(uint32_t));

    start = clock();

    scatter<<<smallNumBlock, BLOCKSIZE>>>(open_count_32_d, open_count_d, open_prefix_sum_d, total_padded_32_div_32, total_padded_32);
    scatter<<<smallNumBlock, BLOCKSIZE>>>(close_count_32_d, close_count_d, close_prefix_sum_d, total_padded_32_div_32, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();

    time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    printf("scatter time: %f\n", time);

    //print_d(open_prefix_sum_d, total_padded_32, ROW1);
    //print_d(close_prefix_sum_d, total_padded_32, ROW1);

    cudaFree(open_count_d);
    cudaFree(close_count_d);
    cudaFree(open_count_32_d);
    cudaFree(close_count_32_d);

    uint32_t* open_block;
    uint32_t* close_block;

    cudaMalloc(&open_block, size*sizeof(uint32_t));
    cudaMalloc(&close_block, size*sizeof(uint32_t));

    start = clock();

    scatter_block<<<numBlock, BLOCKSIZE>>>(open_prefix_sum_d, open_d, open_block, total_padded_32, size);
    scatter_block<<<numBlock, BLOCKSIZE>>>(close_prefix_sum_d, close_d, close_block, total_padded_32, size);
    cudaDeviceSynchronize();

    end = clock();
    time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    printf("scatter block time: %f\n", time);

    //print_d(open_block, size, ROW1);
    //print_d(close_block, size, ROW1);
    cudaFree(open_d);
    cudaFree(close_d);
    cudaFree(open_prefix_sum_d);
    cudaFree(close_prefix_sum_d);
    
    thrust::transform(thrust::cuda::par, open_block, open_block+size, close_block, open_block, thrust::minus<uint32_t>());

    //print_d(open_block, size, ROW1);

    uint32_t* index_d;

    cudaMalloc(&index_d, size*sizeof(uint32_t));
    ///TODO Fill from 0 to N-1
    thrust::sequence(thrust::cuda::par, index_d, index_d+size);

    //print_d(index_d, size, ROW1);


    thrust::transform_if(thrust::cuda::par, index_d, index_d+size, open_block, index_d, functor<set_to_zero>(), not_zero());

    //print_d(index_d, size, ROW1);
    
    int sum_zero = thrust::count_if(thrust::cuda::par, index_d, index_d+size, not_zero());
    //printf("%d\n", sum_zero);
    uint32_t* open_close_reduced_d;
    uint32_t* open_close_reduced_out_d;

    cudaMalloc(&open_close_reduced_d, sum_zero*sizeof(uint32_t));
    cudaFree(open_block);
    cudaFree(close_block);

    thrust::copy_if(thrust::cuda::par, index_d, index_d+size, open_close_reduced_d, not_zero());

    //print_d(open_close_reduced_d, sum_zero, ROW1);
    //open_close_reduced = (uint32_t *) malloc(sizeof(uint32_t)*sum_zero);
    open_close_reduced_size = sum_zero;
    //printf("%d\n", sum_zero);
    open_close_reduced_out_d = open_close_reduced_d;
    //cudaMemcpy(open_close_reduced, open_close_reduced_d, sizeof(uint32_t)*sum_zero, cudaMemcpyDeviceToHost);
    //print(open_close_reduced, sum_zero, ROW1);
    //cudaFree(open_close_reduced_d);
    cudaFree(index_d);
    
    cudaMemcpy(&last_index, open_close_reduced_out_d+sum_zero-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //last_index = open_close_reduced[sum_zero-1];
    if(last_index != size) last_index = last_index+1;
    //printf("%d\n", last_index);
    //last_index = open_close_reduced_d[sum_zero-1];
    //printf("DFSFDSFSD\n");
    printf("*************************************\n");
    return open_close_reduced_out_d;
}


uint8_t* multi_to_one_record( uint8_t* tokens_d, uint32_t last_index_tokens){
    clock_t start, end;
    start = clock();

    int numBlock = ((last_index_tokens) + BLOCKSIZE - 1) / BLOCKSIZE;

    uint32_t res_size = last_index_tokens+3;
    //uint8_t* res = (uint8_t*)malloc(sizeof(uint32_t)*(res_size));
    uint8_t* res_d;
    cudaMalloc(&res_d, sizeof(uint8_t)*res_size);
    //TODO put everything to device array;
    //uint8_t* tokens_d;
    //uint32_t* complete_records_d;
    //cudaMalloc(&tokens_d, sizeof(uint8_t)*ret_size);
    //cudaMalloc(&complete_records_d, sizeof(uint32_t)*open_close_reduced_size_tokens);
    //cudaMemcpy(tokens_d, tokens, sizeof(uint8_t)*ret_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(complete_records_d, complete_records, sizeof(uint32_t)*open_close_reduced_size_tokens, cudaMemcpyHostToDevice);

    //print8(tokens, ret_size, ROW1);
    //print_d(complete_records_d, open_close_reduced_size_tokens, ROW1);
    //printf("%d\n", ret_size);
    //printf("%d\n", res_size);

    parallel_copy<<<numBlock, BLOCKSIZE>>>(tokens_d, res_d, res_size, last_index_tokens);
    
    cudaDeviceSynchronize();

    //cudaMemcpy(res, res_d, res_size*sizeof(uint8_t), cudaMemcpyDeviceToHost);

    end = clock();


    //cudaFree(tokens_d);
    
    //cudaFree(complete_records_d);

    //cudaFree(res_d);

    //printf("inside %f\n", ((double)(end-start)/CLOCKS_PER_SEC)*1000);

    //print8_d<uint8_t>(res_d, res_size, ROW1);
    return res_d;

}


long start(uint8_t * block, uint64_t size, int bLoopCompleted, long* res, double & total_runtime){
    if(bLoopCompleted == 1) return 0;
    uint8_t * block_d;
    uint64_t * parse_tree; 
    uint8_t* tokens_d;
    clock_t start, end;
    double runtime=0, utf_runtime = 0, tokenize_runtime = 0, last_record_runtime = 0, multi_to_one_runtime = 0, parser_runtime=0;
    //printf("%c\n", (char)block[0]);
    cudaMalloc(&block_d, BUFSIZE*sizeof(uint8_t));
    cudaMemcpy(block_d, block, sizeof(uint8_t)*size, cudaMemcpyHostToDevice);
    start = clock();
    bool isValidUTF8 = UTF8Validate(block_d, size);
    end = clock();
    utf_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;

    if(!isValidUTF8) {
        printf("not a valid utf input\n"); 
        exit(0);
    }

    //uint32_t* complete_records;
    uint32_t last_index_tokens;
    //uint32_t open_close_reduced_size_tokens;

    start = clock();
    int ret_size = 0;
    //TODO Pass everything between blocks. in string, 
    uint8_t* in_string_8_d;
    tokens_d = Tokenize(block_d, size, ret_size, in_string_8_d, last_index_tokens);
    end = clock();
    tokenize_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    uint32_t last_index;
    uint32_t open_close_reduced_size;


    //print8_d<uint8_t>(in_string_8_d, size, ROW1);

    start = clock();

    //printf("last_index2: %d\n", last_index_tokens);       // is it possible?

    end = clock();
    last_record_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;

    uint32_t* parser_input;

    start = clock();
    int all_in_one_size = last_index_tokens+3;
    uint8_t* all_in_one_d =  multi_to_one_record(tokens_d, last_index_tokens);
    end = clock();
    multi_to_one_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;

    //int i = last_index;
    //while(tokens[i] != in_string_8[last_index]) i--;
    //i++;
    //printf("i: %d\n", last_index);
    //printf("i: %c\n", in_string_8[i]);
    

    //print8_d<uint8_t>(all_in_one_d, all_in_one_size, ROW1);

    start = clock();
    NewRuntime_Parallel_GPU((char *)all_in_one_d, all_in_one_size);
    end = clock();
    parser_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
                                                        // RUN time ?!
    //print8_d<uint8_t>(all_in_one_d, all_in_one_size, ROW1);
    runtime = utf_runtime + tokenize_runtime + last_record_runtime + multi_to_one_runtime + parser_runtime;

    cudaDeviceSynchronize();
    cudaFree(tokens_d);
    cudaFree(block_d);
    //cudaFree(complete_records_d);
    cudaFree(all_in_one_d);
    cudaFree(in_string_8_d);

    /*
    size_t total, free, allocated;
    cudaMemGetInfo(&free, &total);
    allocated = total - free;
    printf("total: %ld, allocated: %ld, free: %ld\n", total, allocated, free);
    */

    printf("utf runtime: %f\n", utf_runtime);
    printf("tokenize runtime: %f\n", tokenize_runtime);
    printf("last record runtime: %f\n", last_record_runtime);
    printf("multi to one runtime: %f\n", multi_to_one_runtime);
    printf("parser runtime: %f\n", parser_runtime);
    printf("total runtime: %f\n", runtime);
    printf("---------------------------------------------------\n");
    total_runtime += runtime;

    return last_index;

    //return 0; //For now
}

long * readFilebyLine(char* name){
    clock_t start_time, end_time;
    unsigned long  bytesread;
    static uint8_t  buf[BUFSIZE];
    int   sizeLeftover=0;
    int   bLoopCompleted = 0;
    long  pos = 0;
    long * res;
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
        //printf("DSDSDSD\n");

        if(total+read > BUFSIZE){
            start(buf, total, bLoopCompleted, res, total_runtime);
            total = 0;
            i = 0;
            memcpy(buf+total, line, sizeof(uint8_t)*read);
            lineLengths[i] = read;
            total = read; //Reset
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
        start(buf, total, bLoopCompleted, res, total_runtime);
    }
    total = 0;
    
    printf("==================== total runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", total_runtime);

    fclose(handle);
    return res;
 
}



long *readFile(char* name){
    clock_t start_time, end_time;
    unsigned long  bytesread;
    static uint8_t  buf[BUFSIZE];
    int   sizeLeftover=0;
    int   bLoopCompleted = 0;
    long  pos = 0;
    long * res;
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
  long* result;
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