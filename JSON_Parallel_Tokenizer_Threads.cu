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
#include "./Query-response/Parse_Query.hpp"

#define        MAXLINELENGTH    134217728   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824// Max record size
                                            //4MB       8MB     32BM    64MB      128MB    192MB     256MB     512MB     768MB       1GB
#define        BUFSIZE          134217728   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824

#define AVGGPUCLOCK 1346000000


enum hypothesis_val {in, out, unknown, fail};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define RUNTIMES 1

#define BLOCKSIZE 128
#define FILESCOUNT 4
#define NAMELENGTH 25

#define CPUTHREADS 1

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

#define DEBUG

struct start_input_t
{
    uint64_t size;
    int res_size;
    static std::atomic<double> total_runtime;
    static std::atomic<double> utf_runtime;
    static std::atomic<double> tokenize_runtime;
    static std::atomic<double> multi_to_one_runtime;
    static std::atomic<double> parser_runtime;
    static std::atomic<double> move_data_runtime_H_D;
    static std::atomic<double> gput_total_runtime;
    static std::atomic<uint32_t> total_tokens;
    static std::atomic<uint32_t> total_result_size;

    uint8_t * block;
    int32_t* res;
};



inline
cudaError_t checkCuda(cudaError_t result)
{
    printf("CUDA CHECK________\n");
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }

#endif
  return result;
}


std::atomic<double> start_input_t::total_runtime(0);
std::atomic<double> start_input_t::utf_runtime(0);
std::atomic<double> start_input_t::tokenize_runtime(0);
std::atomic<double> start_input_t::multi_to_one_runtime(0);
std::atomic<double> start_input_t::parser_runtime(0);
std::atomic<double> start_input_t::move_data_runtime_H_D(0);
std::atomic<double> start_input_t::gput_total_runtime(0);
std::atomic<uint32_t> start_input_t::total_tokens(0);
std::atomic<uint32_t> start_input_t::total_result_size(0);



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



int print_array(int32_t* input, int length, int rows){
    for(long i =0; i<rows; i++){
      for(long j=0; j<length&&j<100; j++){
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
      for(long j=0; j<100; j++){
        std::bitset<32> y(*(input+j+(i*length)));
        if(j == 129) printf("----129----");
        std::cout << y << ' ';
      }
      std::cout << std::endl;
    }
    free(input);
    return 1;
}

int print8(uint8_t* input, int length, int rows){
    for(long i =0; i<rows; i++){
        for(long j=0; j<length && j<200; j++){
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
        for(long j=0; j<300; j++){
            std::cout << (T )*(input+j+(i*length)) << ' ';
        }
        std::cout << std::endl;
    }
    free(input);
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

            output[i] = input1[i] ^ input2[i];
    }
}

__device__ __forceinline__
void must_be_2_3_continuation_parallel_and_parallel_xor(uint32_t prev1, uint32_t prev2, uint32_t sc, uint32_t& must32_80_sc, uint64_t size, int total_padded_32){
    static const uint32_t third_subtract_byte = (0b11100000u-1) | (0b11100000u-1) << 8 | (0b11100000u-1) << 16 | (0b11100000u-1) << 24;
    static const uint32_t fourth_subtract_byte = (0b11110000u-1) | (0b11110000u-1) << 8 | (0b11110000u-1) << 16 | (0b11110000u-1) << 24;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t is_third_byte = __vsubus4(prev1, third_subtract_byte);
    uint32_t is_fourth_byte = __vsubus4(prev2, fourth_subtract_byte);
    uint32_t gt = __vsubss4((int32_t)(is_third_byte | is_fourth_byte), int32_t(0));
    gt = gt & 0xFFFFFFFF;
    uint32_t must32 = __vcmpgtu4(gt, 0);
    uint32_t must32_80 = must32 & 0x80808080;
    //if(index == 8569) printf("3: %x\n", must32_80);
    must32_80_sc = must32_80 ^ sc;
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
    uint32_t shr_prev1 = (prev1 >> 4) & 0x0f0f0f0f;
    uint32_t shl_prev1 = prev1 & 0x0f0f0f0f;
    uint32_t byte_1_high = (__vcmpltu4(shr_prev1, 0x08080808) & TOO_LONG_32) | (__vcmpgeu4(shr_prev1, 0x08080808) & __vcmpltu4(shr_prev1, 0x0C0C0C0C) & TWO_CONTS_32) | (__vcmpgeu4(shr_prev1, 0x0C0C0C0C) & TOO_SHORT_32) 
    | (__vcmpeq4(shr_prev1, 0x0C0C0C0C) & OVERLONG_2_32) | (__vcmpeq4(shr_prev1, 0x0E0E0E0E) & (OVERLONG_3_32 | SURROGATE_32)) | (__vcmpeq4(shr_prev1, 0x0F0F0F0F) & (TOO_LARGE_32 | TOO_LARGE_1000_32 | OVERLONG_4_32));

    uint32_t byte_1_low = CARRY_32 | (__vcmpltu4(shl_prev1, 0x02020202) & OVERLONG_2_32) | (__vcmpgeu4(shl_prev1, 0x04040404) & TOO_LARGE_32) | (__vcmpgtu4(shl_prev1, 0x04040404) & TOO_LARGE_1000_32) 
            | (__vcmpeq4(shl_prev1, 0) & (OVERLONG_3_32 | OVERLONG_4_32)) | (__vcmpeq4(shl_prev1, 0x0D0D0D0D) & SURROGATE_32);
    
    uint32_t block_compressed_high = (block_compressed >> 4) & 0x0F0F0F0F;
    uint32_t less_than_12 = __vcmpltu4(block_compressed_high, 0x0C0C0C0C);

    uint32_t byte_2_high = ((__vcmpltu4(block_compressed_high, 0x08080808) | __vcmpgtu4(block_compressed_high, 0x0B0B0B0B)) & TOO_SHORT_32) | (less_than_12 & __vcmpgeu4(block_compressed_high, 0x08080808) & (TOO_LONG_32 | OVERLONG_2_32 | TWO_CONTS_32)) 
            | (less_than_12 & __vcmpgtu4(block_compressed_high, 0x08080808) & TOO_LARGE_32) | (__vcmpeq4(block_compressed_high, 0x08080808) & (TOO_LARGE_1000_32 | OVERLONG_4_32)) 
            | (__vcmpgtu4(block_compressed_high, 0x09090909) & less_than_12 & SURROGATE_32); 


    sc =   (byte_1_high & byte_1_low & byte_2_high);
}


__device__ __forceinline__
void prev(uint32_t current, uint32_t previous, uint32_t& prev1, uint32_t& prev2, uint32_t& prev3, uint64_t size, int total_padded_32){

    uint64_t dist = ( ((uint64_t)current) << 32) | (uint64_t)previous;

    prev1 = (uint32_t)(dist >> 3*8);
    prev2 = (uint32_t)(dist >> 2*8);
    prev3 = (uint32_t)(dist >> 1*8);
    // uint32_t test = i!=0 ?  block_d[prev+2] | (block_d[prev+3] << 8) | (first << 16) | (second << 24) : //
    //                                 0 | 0 << 8 | (first << 16) | (second << 24);
    // if(test != permutation_output_d[i]) printf("THERE IS AN ERROR IN permute_32 !!!\n");

}

__global__ 
void is_incomplete(uint32_t* block_compressed_d, uint64_t size, int total_padded_32, bool* one_utf8, int WORDS){
    int tid = threadIdx.x;
    __shared__ uint32_t shared_flag;
    if(tid == 0) shared_flag = 0;
    __syncthreads();
    // if(tid==0)printf("f : %d\n", shared_flag);
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride)
    {
        // unsigned int start_t=0, end_t=0;
        // int start = i*32;
        int start = i*WORDS;
        // start_t = clock();
        // uint32_t first = start<size ? (uint32_t) block[start] : 0;
        // uint32_t second = (start+1)<size ? (uint32_t) block[start+1] : 0;
        // uint32_t third = (start+2)<size ? (uint32_t) block[start+2] : 0;
        // uint32_t fourth = (start+3)<size ? (uint32_t) block[start+3] : 0;
        // end_t = clock();

        #pragma unroll
        for(int j=start; j<size && j<start+WORDS; j++){
            // if (j==size-2)printf("size-2: %x\n", block_compressed_d[j]);

            // if (j==size-1)printf("size-1: %x\n", block_compressed_d[j]);

            uint32_t val = block_compressed_d[j];
            if((val & 0x80808080) != 0) atomicOr(&shared_flag, 1);

        }
        __syncthreads();
        // if(i==8880)printf("clock: %f\n", ((double)(end_t-start_t))/AVGGPUCLOCK * 1000);

        // start_t = clock();

        // uint32_t val = block_compressed_d[i];
        // block_compressed_d[i] = val;
        // prev_incomplete_d[i] = __vsubus4(val, max_val);
        //is_ascii_d[i] =  ((val & 0x80808080) == 0);
        // if((val & 0x80808080) != 0) atomicOr(&shared_flag, 1);

        //bool reg_flag = shared_flag;
        // if(tid==0)printf("%d: %d\n", i, shared_flag);

        // end_t = clock();
        // if(i==8880)printf("clock2: %f\n", ((double)(end_t-start_t))/AVGGPUCLOCK * 1000);

    }
    //printf("%d\n", tid);
    //printf("%d\n", shared_flag);
    if(tid == 0 && shared_flag) *one_utf8 = true;

}

__global__
void check_incomplete_ascii(uint32_t* block_compressed_d, uint32_t* error_d, uint64_t size, int total_padded_32, int WORDS){
    static const uint32_t max_val = (uint32_t)(0b11000000u-1 << 24) | (uint32_t)(0b11100000u-1 << 16) | (uint32_t)(0b11110000u-1 << 8) | (uint32_t)(255); 
    int tid = threadIdx.x;
    __shared__ uint32_t shared_error;
    if(tid == 0) shared_error = 0;
    __syncthreads();
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride)
    {
        int start = i*WORDS;
        #pragma unroll
        for(int j=start; j<size && j<start+WORDS; j++){
            uint32_t current = block_compressed_d[j];
            uint32_t previous = j>0 ? block_compressed_d[j-1] : 0;
            uint32_t prev_incomplete = __vsubus4(previous, max_val);
            //if(i ==0) printf("%ld\n", size);
            //if ((j*WORDS)+255 >= size) printf("last index: %d\n", j);
            if((current & 0x80808080) == 0) {
                atomicExch(&shared_error, prev_incomplete);
                //shared_error = prev_incomplete;
                //error_d[i] =  i > 0 ? prev_incomplete : 0;
            }
            else{
                //uint32_t current = block_compressed_d[i];
                //uint32_t previous = i>0 ? block_compressed_d[i-1] : 0;
                uint32_t prev1, prev2, prev3;
                uint32_t sc;
                uint32_t must32_80_sc;
                prev(current, previous, prev1, prev2, prev3, size, total_padded_32);
                //if(i==8569)printf("1: %x, %x, %x\n", prev1, prev2, prev3);
                check_special_cases(current, prev1, sc, size, total_padded_32);
                //if(i==8569)printf("2: %x\n", sc);
                must_be_2_3_continuation_parallel_and_parallel_xor(prev2, prev3, sc, must32_80_sc, size, total_padded_32);
                //if(i==8569)printf("4: %x\n", must32_80_sc);
                //shared_error = must32_80_sc;
                atomicExch(&shared_error, must32_80_sc);
    
                //if(error_d[i]!=0)printf("I: %x\n", i);
            }
    
    

        }
    }
    __syncthreads();
    if(tid==0 && shared_error) *error_d = shared_error;
}

// __global__
// void do_nothing(uint64_t* input, uint64_t* output, uint64_t size, int total_padded_32){
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for(long i = index; i< total_padded_32; i+=stride)
//     {
//         // if(input[i]) {output[i] =  i > 0 ? input[i-1] : 0;}
//         // else{
//         //     output[i] = input[i];

//         // }
//         // int start = i*16;
//         // for(int j = start; j<total_padded_32 && j<start+16; j++) 
//         output[i] = input[i];

//     }

// }


inline uint8_t prefix_or(uint32_t* is_ascii_d, uint64_t size, int total_padded_32){

    int error = thrust::reduce(thrust::cuda::par, is_ascii_d, is_ascii_d + total_padded_32);
    //printf("%d\n", error);
    return (uint8_t)error;
}
  

inline bool UTF8Validate(uint32_t * block_d, uint64_t size){
    int total_padded_32 = size ;
    uint32_t* general_ptr;
    uint32_t* is_ascii_d;
    uint32_t* prev_incomplete_d;
    uint32_t* block_compressed_d;
    uint32_t* error_d;
    bool one_utf8 = false;
    bool* one_utf8_d;
    cudaMallocAsync(&one_utf8_d, sizeof(bool), 0);
    cudaMemsetAsync(one_utf8_d, 0, sizeof(bool), 0);
    int numBlock = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    // cudaEventRecord(gpu_start);
    //cudaEventSynchronize(gpu_stop);
    //cudaEventElapsedTime(&utf_runtime, gpu_start, gpu_stop);

    ///***  TEST
    // int total_padded_64 = ((size+3)/4);
    // int numBlock_64 = (total_padded_64+BLOCKSIZE-1) / BLOCKSIZE;

    // int total_padded_1024 = ((size+7)/8);
    // int numBlock_1024 = (total_padded_64+BLOCKSIZE-1) / BLOCKSIZE;

    int total_padded_8B = (size+1)/2;
    int total_padded_16B = (size+3)/4;
    int total_padded_24B = (size+5)/6;
    int total_padded_32B = (size+7)/8;
    int total_padded_64B = (size+15)/16;
    int total_padded_128B = (size+31)/32;
    int total_padded_256B = (size+63)/64;
    int total_padded_512B = (size+127)/128;
    int total_padded_1024B = (size+255)/256;
    int total_padded_2048B = (size+511)/512;
    int total_padded_4096B = (size+1023)/1024;

    // printf("%d\n", total_padded_32);
    // printf("%d\n", total_padded_8B);
    // printf("%d\n", total_padded_16B);
    // printf("%d\n", total_padded_24B);
    // printf("%d\n", total_padded_32B);
    // printf("%d\n", total_padded_64B);
    // printf("%d\n", total_padded_128B);
    // printf("%d\n", total_padded_256B);
    // printf("%d\n", total_padded_512B);
    // printf("%d\n", total_padded_1024B);
    // printf("%d\n", total_padded_2048B);
    // printf("%d\n", total_padded_4096B);


    int WORDS = 4;

    int numBlock_8B = (total_padded_8B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_16B = (total_padded_16B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_24B = (total_padded_24B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_32B = (total_padded_32B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_64B = (total_padded_64B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_128B = (total_padded_128B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_256B = (total_padded_256B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_512B = (total_padded_512B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_1024B = (total_padded_1024B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_2048B = (total_padded_2048B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_4096B = (total_padded_4096B+BLOCKSIZE-1) / BLOCKSIZE;
        
    // uint64_t* p1;
    // uint64_t* p2;
    // cudaMallocAsync(&p1, sizeof(uint64_t)*total_padded_64, 0);
    // cudaMallocAsync(&p2, sizeof(uint64_t)*total_padded_64, 0);


    ///****    
    float check_special_cases_runtime = 0;

    cudaMallocAsync(&general_ptr, sizeof(uint32_t), 0);
    //cudaMallocAsync(&block_compressed_d, total_padded_32*sizeof(uint32_t), 0);
    block_compressed_d = block_d;
    error_d = general_ptr;
    // cudaMallocAsync(&is_ascii_d, sizeof(uint32_t)*total_padded_32, 0);
    // cudaMallocAsync(&prev_incomplete_d, sizeof(uint32_t)*total_padded_32, 0);
    // cudaMallocAsync(&block_compressed_d, sizeof(uint32_t)*total_padded_32, 0);
    // cudaMallocAsync(&error_d, sizeof(uint32_t)*total_padded_32, 0);

    // cudaEventRecord(gpu_start);

    //cudaMemcpyAsync(block_compressed_d, block_d, total_padded_32*sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
    // cudaEventRecord(gpu_start);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("incomplete: %f\n", check_special_cases_runtime);

    cudaMemsetAsync(error_d, 0, sizeof(uint32_t), 0);
    cudaEventRecord(gpu_start);
    is_incomplete<<<numBlock_16B, BLOCKSIZE>>>(block_compressed_d, size, total_padded_16B, one_utf8_d, WORDS);
    cudaStreamSynchronize(0);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("incomplete: %f\n", check_special_cases_runtime);
    cudaMemcpyAsync(&one_utf8, one_utf8_d, sizeof(bool), cudaMemcpyDeviceToHost, 0);
    cudaFreeAsync(one_utf8_d, 0);
    //printf("%d\n", one_utf8);
    if(!one_utf8){ 
        cudaFreeAsync(general_ptr, 0);
        //cudaFreeAsync(block_compressed_d, 0);
        return true;
    }
    // cudaMemsetAsync(p1, 1, sizeof(uint64_t)*total_padded_64, 0);
    // cudaEventRecord(gpu_start);
    // do_nothing<<<numBlock_64, BLOCKSIZE>>>(p1, p2, size, total_padded_64);
    // cudaStreamSynchronize(0);
    // cudaEventRecord(gpu_stop);
    // cudaEventSynchronize(gpu_stop);
    // cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("nothing: %f\n", check_special_cases_runtime);
    // cudaFreeAsync(p1,0);
    // cudaFreeAsync(p2,0);

    //exit(0);

    cudaEventRecord(gpu_start);
    check_incomplete_ascii<<<numBlock_16B, BLOCKSIZE>>>(block_compressed_d, error_d, size, total_padded_16B, WORDS);
    cudaStreamSynchronize(0);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&check_special_cases_runtime, gpu_start, gpu_stop);
    // printf("special: %f\n", check_special_cases_runtime);


    uint32_t error = 0;
    cudaMemcpyAsync(&error, error_d, sizeof(uint32_t), cudaMemcpyDeviceToHost, 0);
    //uint8_t error = prefix_or(error_d, size, total_padded_32);
    cudaFreeAsync(general_ptr, 0);
    //cudaFreeAsync(block_compressed_d, 0);
    if(error != 0){ printf("Incomplete ASCII!\n"); return false;}
    return true;

}

__global__
void parallel_or_not_not_and_and_shift_right_shift_left_or_not_and(uint32_t* op_d, uint32_t* whitespace_d, uint32_t* in_string_d, uint32_t* follows_nonquote_scalar_d, uint64_t size, int total_padded_32, int WORDS){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride){

        int start = i*WORDS;
        // start_t = clock();
        // uint32_t first = start<size ? (uint32_t) block[start] : 0;
        // uint32_t second = (start+1)<size ? (uint32_t) block[start+1] : 0;
        // uint32_t third = (start+2)<size ? (uint32_t) block[start+2] : 0;
        // uint32_t fourth = (start+3)<size ? (uint32_t) block[start+3] : 0;
        // end_t = clock();

        #pragma unroll
        for(int k=start; k<size && k<start+WORDS; k++){
            uint32_t op = op_d[k];
            uint32_t whitespace = whitespace_d[k];
            uint32_t in_string = in_string_d[k];

            uint32_t scalar = op | whitespace;
            //scalar = ~scalar;
            in_string = ~in_string;
            //uint32_t nonquote_scalar = scalar & in_string;
            in_string = in_string & scalar;
            //uint32_t overflow = nonquote_scalar >> 31;
            //uint32_t follows_nonquote_scalar = nonquote_scalar << 1;
            //follows_nonquote_scalar = follows_nonquote_scalar | overflow;
            in_string_d[k] = in_string;
            //follows_nonquote_scalar_d[k] = follows_nonquote_scalar;
        }
    }

}

__global__
void get(uint8_t* block_d, uint32_t* output1, uint32_t* output2, uint32_t* op_d, uint32_t* whitespace_d, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        
        int start = i*32;
        uint32_t res1 = 0;
        uint32_t res2 = 0;
        uint32_t res_op = 0;
        uint32_t res_wt = 0;

        for (int j = start; j<start+32 && j<size; j++){

            uint8_t block = block_d[j];
            uint8_t block_low = block & 0x08;
            block == '\\' ? res1 |= 1 << (j-start) : NULL;

            block == '\"' ? res2 |= 1 << (j-start) : NULL;
            //if (block_low != 8) continue;
            res_op |= (((block == '{' ||
                    block == '[' ||
                    block == '}' ||
                    block == ']' ||
                    block == ':' ||
                    block == ','
                    ) ? 1 : 0) << (j-start)) ;
            res_wt |= (((//block == ' ' ||
                    //block == '\t' ||
                    block == '\n'
                    //block == '\r'
                    ) ? 1 : 0) << (j-start)) ;

        }
        output1[i] = res1;
        output2[i] = res2;
        op_d[i] = res_op;
        whitespace_d[i] = res_wt;

    }
}

__global__
void split(uint32_t* input, uint8_t* output, uint64_t size, int total_padded_32){
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
void classify(uint8_t* block_d, uint32_t* op_d, uint32_t* whitespace_d, uint64_t size, int total_padded_32){
    int tid = threadIdx.x;
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
void find_escaped(uint32_t* backslashes_d, uint32_t* quote_d, uint32_t* real_quote_d, int size, int total_padded_32, int WORDS){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {

        int start = i*WORDS;
        // start_t = clock();
        // uint32_t first = start<size ? (uint32_t) block[start] : 0;
        // uint32_t second = (start+1)<size ? (uint32_t) block[start+1] : 0;
        // uint32_t third = (start+2)<size ? (uint32_t) block[start+2] : 0;
        // uint32_t fourth = (start+3)<size ? (uint32_t) block[start+3] : 0;
        // end_t = clock();

        #pragma unroll
        for(int k=start; k<size && k<start+WORDS; k++){
            uint32_t has_overflow = 2;
            uint32_t even_bits = 0x55555555UL;
            long j=k-1;
            if(k==0) has_overflow = 0;
            while(has_overflow == 2){
                uint32_t j_backslash = backslashes_d[j];
                uint32_t follows_escape_t = j_backslash << 1;
                uint32_t odd_seq_t = j_backslash & ~even_bits & ~follows_escape_t;
                uint32_t last_zero = ~(j_backslash | odd_seq_t);
                uint32_t last_one = j_backslash & odd_seq_t;
                uint32_t last_two_bits = (j_backslash & 0xC0000000UL) >> 30;

                has_overflow = (last_two_bits == 2 || (last_two_bits == 3 && last_one>last_zero)) ? 1 : ((last_two_bits == 3 && last_one==last_zero) ? 2 : 0);
                j--;
            }
            uint32_t backslashes = backslashes_d[k];
            backslashes &= ~has_overflow;
            uint32_t follows_escape = (backslashes << 1) | has_overflow;
            uint32_t odd_seq = backslashes & ~even_bits & ~follows_escape;
            uint32_t sequence_starting_even_bits = odd_seq + backslashes;
            uint32_t invert_mask = sequence_starting_even_bits << 1;
            uint32_t escaped = (even_bits ^ invert_mask) & follows_escape;

            uint32_t not_escaped = ~escaped;
            real_quote_d[k] = not_escaped & quote_d[k];
            
            //escaped_d[i] = (even_bits ^ invert_mask) & follows_escape;
            
        }
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
void parallel_copy(uint8_t* tokens_d, uint32_t* tokens_index_d, uint8_t* res_d, uint32_t* res_index_d, uint32_t res_size, uint32_t last_index_tokens){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(uint32_t i = index; i< last_index_tokens; i+=stride)
    {
        if(i == 0){
            uint32_t last_actual_index = tokens_index_d[last_index_tokens-1];
            res_d[0] = '[';
            res_index_d[0] = 0;
            res_d[res_size-2] = ']';
            res_index_d[res_size-2] = last_actual_index+1;
            res_d[res_size-1] = ',';
            res_index_d[res_size-1] = last_actual_index+2;

        }
        res_d[i+1] = tokens_d[i];
        if(tokens_d[i] == '\n' && i < last_index_tokens - 1){
            res_d[i+1] = ',';
            // printf("replaced: %c\n", res_d[i+1]);
        }
        res_index_d[i+1] = tokens_index_d[i]+1;
        
    }
}

__global__
void count_set_bits(uint32_t* input, uint32_t* total_bits, int size, uint32_t total_padded_32, int WORDS){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(uint32_t i = index; i< total_padded_32; i+=stride){
        //UPdate
        int start = i*WORDS;
        #pragma unroll
        for(int k=start; k<size && k<start+WORDS; k++){
            int total = __popc(input[k]);
            total_bits[k] = (uint32_t) total;
        }
    }
}

__global__
void remove_and_copy(uint32_t* set_bit_count, uint32_t* in_string, uint8_t* block_d, uint8_t* in_string_8_d, uint32_t* in_string_8_index_d, uint32_t size, uint32_t total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(uint32_t i = index; i< total_padded_32; i+=stride){
        uint32_t total_before = i > 0 ? set_bit_count[i-1] : 0;
        uint32_t current_total = 0;
        for(int j = 0; j<32 && i*32+j<size; j++){
            uint8_t current_bit = (in_string[i] >> j) & 1;
            current_bit == 1 ? in_string_8_d[total_before+current_total] = block_d[i*32+j] : NULL;
            current_bit == 1 ? in_string_8_index_d[total_before+current_total] = i*32+j : NULL;
            current_total += current_bit;
            //if(current_bit == 1 && block_d[i*32+j] == '\n') printf("new line found\n");

        }  
    }
}

inline uint8_t * Tokenize(uint8_t* block_d, uint64_t size, int &ret_size, uint32_t &last_index_tokens, uint32_t* &in_string_out_index_d){
    int total_padded_32 = (size+31)/32 ;
    int numBlock = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlockBySize = (size + BLOCKSIZE - 1) / BLOCKSIZE;
    uint32_t* backslashes_d;
    //print8_d<uint8_t>(block_d, size, ROW1);

    uint32_t* quote_d;

    uint32_t* general_ptr;

    cudaMallocAsync(&general_ptr, total_padded_32*sizeof(uint32_t)*ROW5, 0);

    clock_t start, end;

    quote_d = general_ptr;
    backslashes_d = general_ptr+total_padded_32;
    uint32_t *whitespace_d, *op_d;
    whitespace_d = general_ptr+total_padded_32*ROW2;
    op_d = general_ptr+total_padded_32*ROW3;


    start = clock();

    get<<<numBlock, BLOCKSIZE>>>(block_d, backslashes_d, quote_d, op_d, whitespace_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    //print8_d<uint8_t>(block_d, (int)size, ROW1);
    //print_d(backslashes_d, total_padded_32, ROW1);
    //print_d(quote_d, total_padded_32, ROW1);
    //print_d(op_d, total_padded_32, ROW1);
    //print_d(whitespace_d, total_padded_32, ROW1);

    end = clock();

    int total_padded_8B = (total_padded_32+1)/2;
    int total_padded_16B = (total_padded_32+3)/4;
    int total_padded_24B = (total_padded_32+5)/6;
    int total_padded_32B = (total_padded_32+7)/8;
    int total_padded_64B = (total_padded_32+15)/16;
    int total_padded_128B = (total_padded_32+31)/32;
    int total_padded_256B = (total_padded_32+63)/64;
    int total_padded_512B = (total_padded_32+127)/128;
    int total_padded_1024B = (total_padded_32+255)/256;
    int total_padded_2048B = (total_padded_32+511)/512;
    int total_padded_4096B = (total_padded_32+1023)/1024;

    // printf("%d\n", total_padded_32);
    // printf("%d\n", total_padded_8B);
    // printf("%d\n", total_padded_16B);
    // printf("%d\n", total_padded_24B);
    // printf("%d\n", total_padded_32B);
    // printf("%d\n", total_padded_64B);
    // printf("%d\n", total_padded_128B);
    // printf("%d\n", total_padded_256B);
    // printf("%d\n", total_padded_512B);
    // printf("%d\n", total_padded_1024B);
    // printf("%d\n", total_padded_2048B);
    // printf("%d\n", total_padded_4096B);


    int WORDS = 2;

    int numBlock_8B = (total_padded_8B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_16B = (total_padded_16B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_24B = (total_padded_24B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_32B = (total_padded_32B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_64B = (total_padded_64B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_128B = (total_padded_128B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_256B = (total_padded_256B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_512B = (total_padded_512B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_1024B = (total_padded_1024B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_2048B = (total_padded_2048B+BLOCKSIZE-1) / BLOCKSIZE;
    int numBlock_4096B = (total_padded_4096B+BLOCKSIZE-1) / BLOCKSIZE;


    // uint32_t* escaped_d;
    // escaped_d = general_ptr+total_padded_32*ROW4;
    uint32_t* real_quote_d;
    real_quote_d = general_ptr+total_padded_32*ROW4;

    start = clock();
    find_escaped<<<numBlock_8B, BLOCKSIZE>>>(backslashes_d, quote_d, real_quote_d, total_padded_32, total_padded_8B, WORDS);
    cudaStreamSynchronize(0);
    //print_d(real_quote_d, total_padded_32, ROW1);
    end = clock();
    // uint32_t* real_quote_d;
    // real_quote_d = general_ptr+total_padded_32;
    // start = clock();
    // parallel_not<<<numBlock, BLOCKSIZE>>>(escaped_d, escaped_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    // start = clock();
    // parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(quote_d, escaped_d, real_quote_d, size, total_padded_32);
    // cudaStreamSynchronize(0);
    // end = clock();
    
    uint32_t* total_one_d;
    total_one_d = general_ptr;
    uint32_t* total_one_32_d;
    start = clock();
    int total_padded_32_div_32 = (total_padded_32+31)/32;

    cudaMallocAsync(&total_one_32_d, (total_padded_32_div_32)*sizeof(uint32_t),0);

    int smallNumBlock = (total_padded_32_div_32 + BLOCKSIZE - 1) / BLOCKSIZE;

    sum_ones<<<smallNumBlock, BLOCKSIZE>>>(real_quote_d, total_one_d, total_one_32_d,  size, total_padded_32, total_padded_32_div_32);
    cudaStreamSynchronize(0);
    end = clock();

    start = clock();
    thrust::exclusive_scan(thrust::cuda::par, total_one_32_d, total_one_32_d+(total_padded_32_div_32), total_one_32_d);
    end = clock();
    uint32_t* prefix_sum_ones;
    //prefix_sum_ones = general_ptr+total_padded_32*ROW3;
    prefix_sum_ones = general_ptr+total_padded_32;

    start = clock();

    scatter<<<smallNumBlock, BLOCKSIZE>>>(total_one_32_d, total_one_d, prefix_sum_ones, total_padded_32_div_32, total_padded_32);
    cudaStreamSynchronize(0);

    end = clock();
    uint32_t* in_string_d;
    in_string_d = general_ptr;
    


    start = clock();
    fact<<<numBlock, BLOCKSIZE>>>(real_quote_d, prefix_sum_ones, in_string_d, total_padded_32_div_32, total_padded_32); // ROW4, ROW0, ROW1
    cudaStreamSynchronize(0);
    cudaFreeAsync(total_one_32_d,0);
    end = clock();

    
    // uint32_t *whitespace_d, *op_d;
    uint32_t* follows_nonquote_scalar_d;

    // op_d = general_ptr+total_padded_32;
    follows_nonquote_scalar_d = general_ptr+total_padded_32*ROW4;
    // whitespace_d = general_ptr+total_padded_32*ROW3;
    // start = clock();
    // classify<<<numBlock, BLOCKSIZE>>>(block_d, op_d, whitespace_d, size, total_padded_32);////////////////////
    // cudaStreamSynchronize(0);
    // end = clock();
    //print_d(in_string_d, total_padded_32, ROW1);
    parallel_or_not_not_and_and_shift_right_shift_left_or_not_and<<<numBlock_8B, BLOCKSIZE>>>(op_d, whitespace_d, in_string_d, follows_nonquote_scalar_d, total_padded_32, total_padded_8B, WORDS);
    //cudaStreamSynchronize(0);
    //print_d(in_string_d, total_padded_32, ROW1);

    uint32_t* set_bit_count;
    set_bit_count = general_ptr+total_padded_32;
    count_set_bits<<<numBlock_8B, BLOCKSIZE>>>(in_string_d, set_bit_count, total_padded_32, total_padded_8B, WORDS);
    cudaStreamSynchronize(0);


    thrust::inclusive_scan(thrust::cuda::par, set_bit_count, set_bit_count+total_padded_32, set_bit_count);
    cudaMemcpyAsync(&last_index_tokens, set_bit_count+total_padded_32-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint8_t* in_string_8_d;
    uint32_t* in_string_8_index_d;
    cudaMallocAsync(&in_string_8_d, last_index_tokens * sizeof(uint8_t),0);
    cudaMallocAsync(&in_string_8_index_d, last_index_tokens * sizeof(uint32_t),0);


    remove_and_copy<<<numBlock, BLOCKSIZE>>>(set_bit_count, in_string_d, block_d, in_string_8_d, in_string_8_index_d, size, total_padded_32);
    cudaStreamSynchronize(0);
    //print8_d<uint8_t>(in_string_8_d, last_index_tokens, ROW1);

    cudaMemcpyAsync(&last_index_tokens, set_bit_count+total_padded_32-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFreeAsync(general_ptr,0);



    in_string_out_index_d = in_string_8_index_d;
    uint8_t* in_string_out_d;
    in_string_out_d = in_string_8_d;
    ret_size = last_index_tokens;

    return in_string_out_d;
}



inline uint8_t* multi_to_one_record( uint8_t* tokens_d, uint32_t* tokens_index_d, uint32_t* &res_index_d, uint32_t last_index_tokens){
    clock_t start, end;
    start = clock();

    int numBlock = ((last_index_tokens) + BLOCKSIZE - 1) / BLOCKSIZE;

    uint32_t res_size = last_index_tokens+3;
    uint8_t* res_d;
    cudaMallocAsync(&res_d, sizeof(uint8_t)*res_size,0);
    cudaMallocAsync(&res_index_d, sizeof(uint32_t)*res_size,0);


    parallel_copy<<<numBlock, BLOCKSIZE>>>(tokens_d, tokens_index_d, res_d, res_index_d, res_size, last_index_tokens);
    
    cudaStreamSynchronize(0);

    end = clock();

    return res_d;

}


inline void * start(void *start_input){
    cudaEvent_t gpu_start_total, gpu_stop_total;
    float gput_total_runtime = 0;
    cudaEventCreate(&gpu_start_total);
    cudaEventCreate(&gpu_stop_total);
    clock_t start, end;
    uint8_t * block = ((start_input_t *)start_input)->block;
    uint64_t size = ((start_input_t *)start_input)->size;
    // uint32_t* res = ((start_input_t *)start_input)->res;
    // double total_runtime = ((start_input_t *)start_input)->total_runtime;
    // double utf_runtime_thead = ((start_input_t *)start_input)->utf_runtime;
    // double tokenize_runtime_thread = ((start_input_t *)start_input)->tokenize_runtime;
    // double multi_to_one_runtime_thread = ((start_input_t *)start_input)->multi_to_one_runtime;
    // double parser_runtime_thread = ((start_input_t *)start_input)->parser_runtime;
    // double move_data_runtime_thread = ((start_input_t *)start_input)->move_data_runtime_H_D;

    // print8(block, size, ROW1);
    float move_data_runtime_H_D = 0;
    //float move_data_runtime_D_H = 0;
    float additional_overhead = 0;
    float additional_overhead_step = 0;


    // uint32_t * block_32 = (uint32_t*)malloc(sizeof(uint32_t)*size_32);
    // uint32_t* new_block_test = (uint32_t *) block;
    // for(int j=0; j<size_32; j++){
    //     int s = j*4;
    //     printf("%x\n", new_block_test[j]);
    //     uint32_t first = s < size ? block[s] : 0;
    //     uint32_t second = s < size ? block[s+1] : 0;
    //     uint32_t third = s < size ? block[s+2] : 0;
    //     uint32_t fourth = s < size ? block[s+3] : 0;
    //     printf("%x %x %x %x\n", first, second, third, fourth);


    //     block_32[j] = ( ( fourth << 24) | (third << 16) | (second << 8) | first );

    // }
    // uint32_t* block_32_d;
    // cudaMallocAsync(&block_32_d, size_32*sizeof(uint32_t), 0);
    
    // start = clock();
    // cudaMemcpyAsync(block_32_d, (uint32_t *)block, size_32*sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
    // end = clock();
    // move_data_runtime += ((float)(end-start)/CLOCKS_PER_SEC)*1000;


    uint8_t * block_d;
    uint64_t * parse_tree; 
    uint8_t* tokens_d;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    int reminder = size%4;
    int padding = (4-reminder)&3;
    uint64_t size_32 = (size + padding)/4;

    size_t limit_v;
    float runtime=0, utf_runtime = 0, tokenize_runtime = 0, last_record_runtime = 0, multi_to_one_runtime = 0, parser_runtime=0;
    cudaEventRecord(gpu_start_total);




    cudaEventRecord(gpu_start);
    cudaMallocAsync(&block_d, (size+padding)*sizeof(uint8_t),0);
    cudaMemsetAsync(block_d, 0, (size+padding)*sizeof(uint8_t), 0);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&additional_overhead_step, gpu_start, gpu_stop);
    additional_overhead += additional_overhead_step;
    // printf("H size-1 %x\n", ((uint32_t *)block)[size_32-1]);
    // printf("H size-1 B %x\n", block[size-1]);
    // printf("H size-2 %x\n", ((uint32_t *)block)[size_32-2]);
    // printf("H size-2 B %x\n", block[size-2]);


    cudaEventRecord(gpu_start);
    cudaMemcpyAsync(block_d, block, sizeof(uint8_t)*size, cudaMemcpyHostToDevice, 0);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&move_data_runtime_H_D, gpu_start, gpu_stop);




    start = clock();
    cudaEventRecord(gpu_start);
    bool isValidUTF8 = UTF8Validate(reinterpret_cast<uint32_t *>(block_d), size_32);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();
    cudaEventElapsedTime(&utf_runtime, gpu_start, gpu_stop);
    //free(block_32);
    //cudaFreeAsync(block_32_d, 0);
    if(!isValidUTF8) {
        printf("not a valid utf input\n"); 
        exit(0);
    }
    //exit(0);
    uint32_t last_index_tokens;


    start = clock();
    cudaEventRecord(gpu_start);
    int ret_size = 0;
    uint32_t* tokens_index_d;
    tokens_d = Tokenize(block_d, size, ret_size, last_index_tokens, tokens_index_d);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();
    cudaEventElapsedTime(&tokenize_runtime, gpu_start, gpu_stop);

    uint32_t last_index;
    uint32_t open_close_reduced_size;
    uint32_t* parser_input;


    printf("---------------------------------------------------\n");
    cudaEventRecord(gpu_stop_total);
    cudaEventSynchronize(gpu_stop_total);
    cudaEventElapsedTime(&gput_total_runtime, gpu_start_total, gpu_stop_total);
    printf("total runtime till here: %f\n", gput_total_runtime);

    start = clock();
    cudaEventRecord(gpu_start);
    uint32_t* all_in_one_index_d;
    int all_in_one_size = last_index_tokens+3;
    uint8_t* all_in_one_d =  multi_to_one_record(tokens_d, tokens_index_d, all_in_one_index_d, last_index_tokens);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();
    cudaEventElapsedTime(&multi_to_one_runtime, gpu_start, gpu_stop);
    // print8_d<uint8_t>(all_in_one_d, all_in_one_size, ROW1);

    cudaFreeAsync(tokens_index_d, 0);    
    cudaFreeAsync(tokens_d,0);
    cudaFreeAsync(block_d,0); ////// 

    int32_t* result_d;
    int32_t* result;
    int result_size;

    start = clock();
    cudaEventRecord(gpu_start);
    result_d = NewRuntime_Parallel_GPU((char *)all_in_one_d, (int32_t **)(&all_in_one_index_d),  all_in_one_size, result_size);
    //result = (int32_t*)malloc(sizeof(int32_t)*result_size*ROW2);
    // printf("res_size: %d\n", result_size);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    end = clock();

    parser_runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    cudaEventElapsedTime(&parser_runtime, gpu_start, gpu_stop);
    //float copy_res_runtime = 0;
    //cudaMallocHost(&((start_input_t *)start_input)->res, sizeof(int32_t)*result_size*ROW2);
    ((start_input_t *)start_input)->res_size = result_size;
    //result = (int32_t *)(((start_input_t *)start_input)->res);
    //cudaEventRecord(gpu_start);
    //cudaMemcpyAsync(result, result_d, sizeof(int32_t)*result_size*ROW2, cudaMemcpyDeviceToHost, 0);
    //cudaEventRecord(gpu_stop);
    //cudaEventSynchronize(gpu_stop);
    //cudaEventElapsedTime(&move_data_runtime_D_H, gpu_start, gpu_stop);
    //printf("copy res runtime: %f\n", copy_res_runtime);

    //cudaFreeAsync(result_d, 0);

    //printf("%c\n", block[0]);
    // printf("%ld\n", size);
    // for(int k = 1 ; k<*result; k++){
    //     // printf("child_node_address: %d\n", *(result+k));
    //     int child_node_address = *(result+k); // node address in array
    //     int num_child_of_child_node = result[child_node_address]; // node child numbers
    //     // printf("num_child_of_child_node: %d\n", num_child_of_child_node);
    //     int child_node_string_index = result[result_size*ROW1+child_node_address]-1;
    //     //printf("child_node_string_index: %d ", child_node_string_index);
    //     //int previous_node_index = k>0 ?  *(result+k-1) : result[0];
    //     printf("%c ", block[child_node_string_index]);
    // }
    // printf("\n");
    // exit(0);
    runtime = utf_runtime + tokenize_runtime + multi_to_one_runtime + parser_runtime;

    cudaStreamSynchronize(0);
    cudaFreeAsync(all_in_one_d,0);
    //cudaFreeHost(result);
    start = clock();
    uint32_t total_tokens = (uint32_t) all_in_one_size;
    uint32_t total_result_size = (uint32_t) result_size*ROW3;

    #if defined (DEBUG)
    printf("total tokens : %d\n", all_in_one_size);
    printf("result size : %d\n\n", result_size*ROW3);
    printf("utf runtime: %f\n", utf_runtime);
    printf("tokenize runtime: %f\n", tokenize_runtime);
    printf("multi to one runtime: %f\n", multi_to_one_runtime);
    printf("parser runtime: %f\n", parser_runtime);
    printf("move data runtime H to D: %f\n", move_data_runtime_H_D);
    //printf("move data runtime D to H: %f\n", move_data_runtime_D_H);
    printf("total runtime: %f\n", runtime);

    #endif

    #if defined (DEBUG)
    for(double temp = ((start_input_t *)start_input)->total_runtime;
        !(((start_input_t *)start_input)->total_runtime).compare_exchange_strong(temp, temp+runtime););
    for(double temp = ((start_input_t *)start_input)->utf_runtime;
        !(((start_input_t *)start_input)->utf_runtime).compare_exchange_strong(temp, temp+utf_runtime););
    for(double temp = ((start_input_t *)start_input)->tokenize_runtime;
        !(((start_input_t *)start_input)->tokenize_runtime).compare_exchange_strong(temp, temp+tokenize_runtime););
    for(double temp = ((start_input_t *)start_input)->multi_to_one_runtime;
        !(((start_input_t *)start_input)->multi_to_one_runtime).compare_exchange_strong(temp, temp+multi_to_one_runtime););
    for(double temp = ((start_input_t *)start_input)->parser_runtime;
        !(((start_input_t *)start_input)->parser_runtime).compare_exchange_strong(temp, temp+parser_runtime););
    for(double temp = ((start_input_t *)start_input)->move_data_runtime_H_D;
        !(((start_input_t *)start_input)->move_data_runtime_H_D).compare_exchange_strong(temp, temp+move_data_runtime_H_D););
    for(double temp = ((start_input_t *)start_input)->gput_total_runtime;
        !(((start_input_t *)start_input)->gput_total_runtime).compare_exchange_strong(temp, temp+gput_total_runtime););
    for(uint32_t temp = ((start_input_t *)start_input)->total_tokens;
        !(((start_input_t *)start_input)->total_tokens).compare_exchange_strong(temp, temp+total_tokens););
    for(uint32_t temp = ((start_input_t *)start_input)->total_result_size;
        !(((start_input_t *)start_input)->total_result_size).compare_exchange_strong(temp, temp+total_result_size););
    #endif
    end = clock();
    additional_overhead += ((float)(end-start)/CLOCKS_PER_SEC)*1000;

    #if defined (DEBUG)
    printf("overhead runtime: %f\n", additional_overhead);
    #endif
    return (void *)result_d;
}


void get_key_value(start_input_t threads_value[], int32_t * all_res){
    //printf("result size : %d\n", (start_input[j].res_size)*ROW2);
    int result_size = threads_value[0].res_size;
    int32_t * res = all_res;
    uint8_t * block = threads_value[0].block;
    for(int k = 1 ; k<*res && k<3; k++){
        // printf("child_node_address: %d\n", *(result+k));
        int child_node_address = *(res+k); // node address in array
        int next_child_node_address = *(res+k+1); // node address in array

        int child_node_string_index = res[result_size*ROW1+k]-1;

        int num_child_of_child_node = res[child_node_address]; // node child numbers
        // printf("num_child_of_child_node: %d\n", num_child_of_child_node);
        int child_node_length = res[result_size*ROW1+child_node_address];
        //int next_child_node_string_index = res[result_size*ROW1+next_child_node_address]-1;
        printf("node address: %d, next node address: %d\n", child_node_address, next_child_node_address);
        printf("node length: %d, start index: %d\n", child_node_length, child_node_string_index);    
        if(k < (*res)-1) printf("%.*s\n", child_node_length, (block)+child_node_string_index);
        //printf("child_node_string_index: %d ", child_node_string_index);
        //int previous_node_index = k>0 ?  *(result+k-1) : result[0];
        //printf("%c ", start_input[j].block[child_node_string_index]);
    }
    printf("\n");
}

inline int32_t * readFilebyLine(char* name){
    clock_t start_time, end_time;
    unsigned long  bytesread;
    //static uint8_t  buf[BUFSIZE];
    static uint8_t*  buf;
    static int32_t* res_buf;
    cudaMallocHost(&buf, sizeof(uint8_t)*BUFSIZE);
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    float move_data_runtime_D_H = 0;
    float total_move_data_runtime_D_H = 0;

    checkCuda(cudaMallocHost(&res_buf, sizeof(uint32_t)*BUFSIZE*ROW3));
    //res_buf = (int32_t*)malloc(sizeof(int32_t)*BUFSIZE*ROW2);
    //buf = (uint8_t*)malloc(sizeof(uint8_t)*BUFSIZE);
    const int CPU_threads_m_one = CPUTHREADS - 1;
    int   sizeLeftover=0;
    long  pos = 0;
    int32_t * res; //= (int32_t **)malloc(sizeof(int32_t *) * CPUTHREADS);
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
            //pthread_t threads[CPUTHREADS];
            start_input_t start_input[CPUTHREADS];
            int sub_input_index = (i+CPU_threads_m_one)/CPUTHREADS;
            start_time = clock();
            start_input[0].block = buf;
            start_input[0].size = lineLengths[sub_input_index*(1)-1];
            res = (int32_t*)start((void *)start_input); // start function call
            //int total_size = 0;
            //pthread_join(threads[j], (void **)(res+j));
            cudaEventRecord(gpu_start);
            cudaMemcpy(res_buf, res, sizeof(int32_t)*(start_input[0].res_size)*ROW3, cudaMemcpyDeviceToHost);
            //(start_input+j)->res = 0; //res_buf+total_size;
            cudaEventRecord(gpu_stop);
            cudaEventSynchronize(gpu_stop);
            cudaEventElapsedTime(&move_data_runtime_D_H, gpu_start, gpu_stop);
            total_move_data_runtime_D_H += move_data_runtime_D_H;
            cudaFree(res);
            res = res_buf;
            //printf("%d\n", *(res[j]));
            //total_size += (start_input[0].res_size)*ROW2;
            
            // for(int j=0; j<CPUTHREADS && j < sub_input_index; j++){
            //     start_input[j].block = j > 0 ? buf+lineLengths[sub_input_index*j-1] : buf;
            //     start_input[j].size = j < CPU_threads_m_one ?  (j > 0 ? 
            //                                 lineLengths[sub_input_index*(j+1)-1] - lineLengths[sub_input_index*j-1] : 
            //                                 lineLengths[sub_input_index*(j+1)-1] ) :
            //                         lineLengths[i-1] - lineLengths[sub_input_index*j-1];
                
            //     if(pthread_create(&threads[j], NULL, start, (void *)& start_input[j])) break;

            // }
            //int total_size = 0;
            //pthread_join(threads[j], (void **)(res+j));
            // cudaEventRecord(gpu_start);
            // cudaMemcpy(res_buf, res, sizeof(int32_t)*(start_input[0].res_size)*ROW2, cudaMemcpyDeviceToHost);
            // //(start_input+j)->res = 0; //res_buf+total_size;
            // cudaEventRecord(gpu_stop);
            // cudaEventSynchronize(gpu_stop);
            // cudaEventElapsedTime(&move_data_runtime_D_H, gpu_start, gpu_stop);
            // total_move_data_runtime_D_H += move_data_runtime_D_H;
            // cudaFree(res);
            // res = res_buf;
            //printf("%d\n", *(res[j]));
            //total_size += (start_input[j].res_size)*ROW2;
            // for(int j=0; j<CPUTHREADS; j++){
            //     pthread_join(threads[j], (void **)(res+j));
            //     cudaEventRecord(gpu_start);
            //     cudaMemcpy(res_buf+total_size, res[j], sizeof(int32_t)*(start_input[j].res_size)*ROW2, cudaMemcpyDeviceToHost);
            //     //(start_input+j)->res = 0; //res_buf+total_size;
            //     cudaEventRecord(gpu_stop);
            //     cudaEventSynchronize(gpu_stop);
            //     cudaEventElapsedTime(&move_data_runtime_D_H, gpu_start, gpu_stop);
            //     total_move_data_runtime_D_H += move_data_runtime_D_H;
            //     cudaFree(res[j]);
            //     res[j] = res_buf+total_size;
            //     //printf("%d\n", *(res[j]));
            //     total_size += (start_input[j].res_size)*ROW2;
            // }
            cudaDeviceSynchronize();
            end_time = clock();
            #if defined(DEBUG)
            std::cout << "-------------DEBUG----------------" <<  std::endl;
            clock_t query_start, query_end;
            double query_time = 0;
            //get_key_value(start_input, res);
            query_start = clock();
            //print_array(res, start_input[0].res_size, ROW3);
            structural_iterator str_iter(res, start_input[0].block, start_input[0].res_size, start_input[0].size);
            //print_array(res, start_input[0].res_size, ROW3);
            //printf("%d, %d, %d\n", res[start_input[0].res_size+ 1]-1, res[1], res[start_input[0].res_size+res[start_input[0].res_size*ROW2+1]]-1);
            //printf("%c, %d, %c\n", start_input[0].block[res[start_input[0].res_size+1]-1], res[1], start_input[0].block[res[start_input[0].res_size+res[start_input[0].res_size*ROW2+1]]-1]);
            int error = 1;
            int index = 0;
            int which_file = 2;
            string json_result;
            string json_key;
            switch(which_file){
                // query for google_map: $[0].routes.[0].overview_polyline.points
                case 0:
                error = str_iter.goto_array_index(1);
                if(!error) exit(0);
                index = str_iter.find_specific_key("routes");
                if(!index) exit(0);
                error =str_iter.goto_index(index);
                if(!error) exit(0);
                error = str_iter.goto_array_index(1);
                if(!error) exit(0);
                index = str_iter.find_specific_key("overview_polyline");
                if(!index) exit(0);
                error = str_iter.goto_index(index);
                if(!error) exit(0);
                index = str_iter.find_specific_key("points");
                if(!index) exit(0);
                error = str_iter.goto_index(index);
                if(!error) exit(0);
                json_result = str_iter.get_value();
                json_key = str_iter.get_key();
                break;
                case 1:
                // query for wiki: $[0].aliases.zh-hant.[1].value
                error = str_iter.goto_array_index(1);
                if(!error) break;
                index = str_iter.find_specific_key("aliases");
                if(!index) break;
                error = str_iter.goto_index(index);
                if(!error) break;
                index = str_iter.find_specific_key("zh-hant");
                if(!index) break;
                error = str_iter.goto_index(index);
                if(!error) exit(0);
                error = str_iter.goto_array_index(2);
                if(!error) exit(0);
                index = str_iter.find_specific_key("value");
                if(!index) exit(0);
                error = str_iter.goto_index(index);
                if(!error) exit(0);
                json_result = str_iter.get_value();
                json_key = str_iter.get_key();
                break;
                case 2:
                // query for nspl: $[0].[1]
                error = str_iter.goto_array_index(3);
                //printf("Done!\n");
                if(!error) exit(0);
                error = str_iter.goto_array_index(2);
                //printf("Done!\n");
                if(!error) exit(0);
                json_result = str_iter.get_value();
                //json_key = str_iter.get_key();
                break;
                case 3:
                // query for twitter: $[0].user.location
                error = str_iter.goto_array_index(1);
                if(!error) exit(0);
                index = str_iter.find_specific_key("user");
                if(!index) exit(0);
                error =str_iter.goto_index(index);
                if(!error) exit(0);
                index = str_iter.find_specific_key("location");
                if(!index) exit(0);
                error = str_iter.goto_index(index);
                if(!error) exit(0);
                json_result = str_iter.get_value();
                json_key = str_iter.get_key();
                break;
                case 4:
                // query for walmart: $[0].salePrice
                error = str_iter.goto_array_index(1);
                if(!error) break;
                index = str_iter.find_specific_key("salePrice");
                if(!index) break;
                error =str_iter.goto_index(index);
                json_result = str_iter.get_value();
                json_key = str_iter.get_key();
                break;
            }
            query_end = clock();
            query_time = ((double)(query_end-query_start)/CLOCKS_PER_SEC)*1000;
            //std::cout << json_result << std::endl;
            std::cout << json_key << ' ' << json_result << std::endl;
            printf("query runtime is: %f\n", query_time);
            #endif

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
        //pthread_t threads[CPUTHREADS];
        start_input_t start_input[CPUTHREADS];
        int sub_input_index = (i+CPU_threads_m_one)/CPUTHREADS;
        start_time = clock();
        start_input[0].block = buf;
        start_input[0].size = lineLengths[sub_input_index*(1)-1];
        
        //if(pthread_create(&threads[j], NULL, start, (void *)& start_input[j])) break;
        res = (int32_t*)start((void *)start_input); // start function call

        // for(int j=0; j<CPUTHREADS && j < sub_input_index; j++){
        //     start_input[j].block = j > 0 ? buf+lineLengths[sub_input_index*j-1] : buf;
        //     start_input[j].size = j < CPU_threads_m_one ?  (j > 0 ? 
        //                                 lineLengths[sub_input_index*(j+1)-1] - lineLengths[sub_input_index*j-1] : 
        //                                 lineLengths[sub_input_index*(j+1)-1] ) :
        //                         lineLengths[i-1] - lineLengths[sub_input_index*j-1];
            
        //     if(pthread_create(&threads[j], NULL, start, (void *)& start_input[j])) break;

        // }
        //int total_size = 0;
        //pthread_join(threads[j], (void **)(res+j));
        cudaEventRecord(gpu_start);
        cudaMemcpy(res_buf, res, sizeof(int32_t)*(start_input[0].res_size)*ROW2, cudaMemcpyDeviceToHost);
        //start_input[j].res = res_buf+total_size;
        cudaEventRecord(gpu_stop);
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&move_data_runtime_D_H, gpu_start, gpu_stop);
        total_move_data_runtime_D_H += move_data_runtime_D_H;
        cudaFree(res);
        res = res_buf;
        //total_size += (start_input[j].res_size)*ROW2;
        //printf("%d\n", *(start_input[j].res));

        // for(int j=0; j<CPUTHREADS && j < sub_input_index; j++){
        //     pthread_join(threads[j], (void **)(res+j));
        //     cudaEventRecord(gpu_start);
        //     cudaMemcpy(res_buf+total_size, res[j], sizeof(int32_t)*(start_input[j].res_size)*ROW2, cudaMemcpyDeviceToHost);
        //     //start_input[j].res = res_buf+total_size;
        //     cudaEventRecord(gpu_stop);
        //     cudaEventSynchronize(gpu_stop);
        //     cudaEventElapsedTime(&move_data_runtime_D_H, gpu_start, gpu_stop);
        //     total_move_data_runtime_D_H += move_data_runtime_D_H;
        //     cudaFree(res[j]);
        //     res[j] = res_buf+total_size;
        //     total_size += (start_input[j].res_size)*ROW2;
        //     //printf("%d\n", *(start_input[j].res));

        // }
        cudaDeviceSynchronize();
        end_time = clock();
        total_runtime += ((double)(end_time-start_time)/CLOCKS_PER_SEC)*1000;
        //start(buf, total, res, total_runtime);
        #if defined(DEBUG)
        //get_key_value(start_input, res);
        #endif
    }
    total = 0;
    
    printf("==================== total runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", total_runtime); //start_input_t::total_runtime.load()
    printf("==================== utf runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::utf_runtime.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== tokenize runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::tokenize_runtime.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== multi to one runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::multi_to_one_runtime.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== parser runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::parser_runtime.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== move data runtime H to D ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::move_data_runtime_H_D.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== move data runtime D to H ====================\n\t\t\t %f\n\
====================      end      ====================\n", total_move_data_runtime_D_H/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== gput total runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::gput_total_runtime.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== total runtime ====================\n\t\t\t %f\n\
====================      end      ====================\n", start_input_t::total_runtime.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== total tokens ====================\n\t\t\t %d\n\
====================      end      ====================\n", start_input_t::total_tokens.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    printf("==================== total result size ====================\n\t\t\t %d\n\
====================      end      ====================\n", start_input_t::total_result_size.load()/CPUTHREADS); //start_input_t::total_runtime.load()
    cudaFreeHost(res_buf);
    cudaFreeHost(buf);
    fclose(handle);
    return res;
 
}




int main(int argc, char **argv)
{
    int32_t* result;
  if (argv[1] != NULL){
    if( strcmp(argv[1], "-b") == 0 && argv[2] != NULL){
      std::cout << "Batch mode..." << std::endl;
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