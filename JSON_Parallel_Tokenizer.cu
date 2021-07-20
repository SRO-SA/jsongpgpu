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
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <string.h>
#include <pthread.h>
#include <bitset>
#include "JSON_Parallel_Parser.h"

#define        MAXLINELENGTH    4194304 // Max record size
#define        BUFSIZE      4194304

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
    input = (uint32_t*) malloc(sizeof(uint32_t)*length);
    cudaMemcpy(input, input_d, sizeof(uint32_t)*length, cudaMemcpyDeviceToHost);

    for(long i =0; i<rows; i++){
      for(long j=0; j<length; j++){
        std::bitset<32> y(*(input+j+(i*length)));
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
void single_parallel_and(T* input, int value, T* output, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        output[i] = input[i] & value; 
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

__global__
void permute_32(uint8_t *block_d, uint32_t *permutation_output_d, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        int prev = (i-1)*4;
        int start = i*4;
        permutation_output_d[i] = block_d[prev+2] | (block_d[prev+3] << 8) | (block_d[start] << 16) | (block_d[start+1] << 24);
    }
}

__global__
void align_right(uint32_t* permutation_output_d, uint8_t* block_d, uint32_t* prev_d, uint64_t size, int total_padded_32, int shift){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //int prev = (i-1)*4;
        int start = i*4;
        uint32_t dist1 = ((block_d[start] << 24 | block_d[start+1] << 16) 
        | (permutation_output_d[i] & 0x0000ffff)) >> shift*8;
        uint32_t dist2 = ((block_d[start+2] << 24 | block_d[start+3] << 16) 
        | ((permutation_output_d[i] & 0xffff0000) >> 16)) >> shift*8;

    
        prev_d[i] = dist1 | (dist2 << 16);
    }

}


__global__ 
void is_incomplete(uint8_t* block, uint32_t* prev_incomplete_d, uint64_t size, int total_padded_32){
    static const uint32_t max_val = (uint32_t)(0b11000000u-1) | (uint32_t)(0b11100000u-1 << 8) | (uint32_t)(0b11110000u-1 << 16) | (uint32_t)(255 << 24);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(long i = index; i< total_padded_32; i+=stride)
    {
        //int prev = (i-1)*4;
        int start = i*4;
        uint32_t val = (uint32_t)block[start] | (uint32_t)(block[start+1] << 8) | (uint32_t)(block[start+2] << 16) | (uint32_t)(block[start+3] << 24);
        prev_incomplete_d[i] = __vsubus4(val, max_val);
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
        is_ascii_d[i] = (((block[start]&0b10000000) >> 7) 
                    | ((block[start+1]&0b10000000) >> 6) 
                    | ((block[start+2]&0b10000000) >> 5) 
                    | ((block[start+3]&0b10000000) >> 4)) == 0;
        if(is_ascii_d[i]) is_ascii_d[i] = prev_incomplete_d[i-1];
        else is_ascii_d[i] = 0;
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
        uint32_t is_third_byte = __vsubus4(prev1[i], 0b11100000u-1);
        uint32_t is_fourth_byte = __vsubus4(prev1[i], 0b11110000u-1);
        uint32_t gt = __vsubus4((is_third_byte | is_fourth_byte), int8_t(0));
        must32_d[i] = ~(gt == uint8_t(0));
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
    constexpr const uint8_t CARRY = TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .
    uint8_t table2[16] = {
          // ____0000 ________
          CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
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
        uint8_t shr_prev1_b1 = shr_prev1;
        uint8_t shr_prev1_b2 = (shr_prev1 >> 8);
        uint8_t shr_prev1_b3 = (shr_prev1 >> 16);
        uint8_t shr_prev1_b4 = (shr_prev1 >> 24);
        uint32_t byte_1_high = table1[shr_prev1_b1] | (table1[shr_prev1_b2] << 8) | (table1[shr_prev1_b3] << 16) | (table1[shr_prev1_b4] << 24);

        uint32_t shl_prev1 = prev1_d[i] & 0x0f0f0f0f;
         shr_prev1_b1 = shl_prev1;
         shr_prev1_b2 = (shl_prev1 >> 8);
         shr_prev1_b3 = (shl_prev1 >> 16);
         shr_prev1_b4 = (shl_prev1 >> 24);

        uint32_t byte_1_low = table2[shr_prev1_b1] | (table2[shr_prev1_b2] << 8) | (table2[shr_prev1_b3] << 16) | (table2[shr_prev1_b4] << 24);

        shr_prev1_b1 = (block_d[start] >> 4);
        shr_prev1_b2 = (block_d[start+1] >> 4);
        shr_prev1_b3 = (block_d[start+2] >> 4);
        shr_prev1_b4 = (block_d[start+3] >> 4);

        uint32_t byte_2_high = table3[shr_prev1_b1] | (table3[shr_prev1_b2] << 8) | (table3[shr_prev1_b3] << 16) | (table3[shr_prev1_b4] << 24);

        sc_d[i] =   (byte_1_high & byte_1_low & byte_2_high);
    }

}


uint8_t prefix_or(uint32_t* is_ascii_d, uint64_t size, int total_padded_32){

    thrust::inclusive_scan(thrust::cuda::par, is_ascii_d, is_ascii_d + total_padded_32, is_ascii_d);
    return (uint8_t)is_ascii_d[total_padded_32-1];

}
  

bool UTF8Validate(uint8_t * block_d, uint64_t size){
    int total_padded_32 = ((size + 3)/4) ;
    uint32_t* is_ascii_d;
    uint32_t* permutation_output_d;
    uint32_t* prev_incomplete_d;
    int numBlock = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;

    print8_d<uint8_t>(block_d, (int)size, ROW1);


    cudaMalloc(&is_ascii_d, sizeof(uint32_t)*total_padded_32);
    cudaMalloc(&prev_incomplete_d, sizeof(uint32_t)*total_padded_32);

    is_incomplete<<<numBlock, BLOCKSIZE>>>(block_d, prev_incomplete_d, size, total_padded_32);

    print_d(prev_incomplete_d, total_padded_32, ROW1);

    check_incomplete_ascii<<<numBlock, BLOCKSIZE>>>(block_d, prev_incomplete_d, is_ascii_d, size, total_padded_32);
    uint8_t error = prefix_or(is_ascii_d, size, total_padded_32);
    std::cout << "HERE" << std::endl;

    if(error != 0) return false;

    cudaMalloc(&permutation_output_d, sizeof(uint32_t)*total_padded_32);
    permute_32<<<numBlock, BLOCKSIZE>>>(block_d, permutation_output_d, size, total_padded_32);
    uint32_t* prev1_d=is_ascii_d, *sc_d=prev_incomplete_d;
    //cudaMalloc(&prev1, sizeof(uint32_t)*total_padded_32);
    //cudaMalloc(&sc, sizeof(uint32_t)*total_padded_32);
    align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev1_d, size, total_padded_32, 4-1);
    check_special_cases<<<numBlock, BLOCKSIZE>>>(block_d, prev1_d, sc_d, size, total_padded_32);
    uint32_t* prev2_d = prev1_d, *prev3_d;
    cudaMalloc(&prev2_d, sizeof(uint32_t)*total_padded_32);
    cudaMalloc(&prev3_d, sizeof(uint32_t)*total_padded_32);
    align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev2_d, size, total_padded_32, 4-2);
    align_right<<<numBlock, BLOCKSIZE>>>(permutation_output_d, block_d, prev3_d, size, total_padded_32, 4-3);
    uint32_t* must32_d;
    cudaMalloc(&must32_d, sizeof(uint32_t)*total_padded_32);
    must_be_2_3_continuation<<<numBlock, BLOCKSIZE>>>(prev2_d, prev3_d, must32_d, size, total_padded_32);
    uint32_t* must32_80_d, *must32_80_sc_d;
    cudaMalloc(&must32_80_d, sizeof(uint32_t)*total_padded_32);
    cudaMalloc(&must32_80_sc_d, sizeof(uint32_t)*total_padded_32);

    single_parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(must32_d, 0x80, must32_80_d, size, total_padded_32);

    parallel_xor<<<numBlock, BLOCKSIZE>>>(must32_80_d, sc_d, must32_80_sc_d, size, total_padded_32);


    error = prefix_or(must32_80_sc_d, size, total_padded_32);
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
            for(int j = 0; j<32 && start+j < size; j++){
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
                            else if((cur_b == ']' || cur_b == '}') && j+start == size-1 ) {state = 0; j=33;}

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
                            if(cur_b == ',' || cur_b == '}' || cur_b == ']') state = 0;
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
                            j = 33;
                        break;}
                    }
                }
                else break;
            }
            run++;
            if (run>1) break;
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
                    block_d[j] == '\n' ||
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
        long j=i;
        while(has_overflow == 2){
            uint32_t follows_escape_t = backslashes_d[j] << 1;
            uint32_t odd_seq_t = backslashes_d[j] & ~even_bits & ~follows_escape_t;
            uint32_t last_zero = ~(backslashes_d[j] | odd_seq_t);
            uint32_t last_one = backslashes_d[j] & odd_seq_t;
            uint32_t last_two_bits = (backslashes_d[j] & 0xC000UL) >> 30;
            has_overflow = last_two_bits == 2 ? 1 : (last_two_bits == 3 & last_one>last_zero) ? 2 : 0;
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

uint8_t * Tokenize(uint8_t* block_d, uint64_t size){
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
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(backslashes_d, total_padded_32, ROW1);
    //printf("%d\n", total_padded_32);

    uint32_t* escaped_d;
    start = clock();
    cudaMalloc(&escaped_d, total_padded_32 * sizeof(uint32_t));
    find_escaped<<<numBlock, BLOCKSIZE>>>(backslashes_d, escaped_d, total_padded_32);
    cudaDeviceSynchronize();
    cudaFree(backslashes_d);
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //printf("ecaped backslashes: \n");
    //print_d(escaped_d, total_padded_32, ROW1);

    uint32_t* quote_d;
    start = clock();
    cudaMalloc(&quote_d, total_padded_32 * sizeof(uint32_t));
    get<<<numBlock, BLOCKSIZE>>>(block_d, quote_d, size, total_padded_32, 1);//////////////
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(quote_d, total_padded_32, ROW1);

    uint32_t* real_quote_d;
    start = clock();
    cudaMalloc(&real_quote_d, total_padded_32 * sizeof(uint32_t));
    parallel_not<<<numBlock, BLOCKSIZE>>>(escaped_d, escaped_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(escaped_d, total_padded_32, ROW1);
    start = clock();
    parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(quote_d, escaped_d, real_quote_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    cudaFree(quote_d);
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(real_quote_d, total_padded_32, ROW1);

    uint32_t* prediction_d;
    start = clock();
    cudaMalloc(&prediction_d, size * sizeof(uint32_t));
    predict<<<numBlock, BLOCKSIZE>>>(block_d, real_quote_d, prediction_d, size, total_padded_32);
    cudaDeviceSynchronize();
    cudaFree(real_quote_d);
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(prediction_d, total_padded_32, ROW1);

    uint32_t* in_string_d;
    cudaMalloc(&in_string_d, total_padded_32 * sizeof(uint32_t));
    cudaMemcpy(in_string_d, prediction_d, sizeof(uint32_t)*total_padded_32, cudaMemcpyDeviceToDevice);
    cudaFree(prediction_d);
    
    /////////
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
    */
    //////////

    ///////////////////////////////////////
    /*

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
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(whitespace_d, total_padded_32, ROW1);
    //print_d(op_d, total_padded_32, ROW1);

    uint32_t* scalar_d;
    start = clock();
    cudaMalloc(&scalar_d, total_padded_32*sizeof(uint32_t));
    parallel_or<<<numBlock, BLOCKSIZE>>>(op_d, whitespace_d, scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(scalar_d, total_padded_32, ROW1);
    start = clock();
    parallel_not<<<numBlock, BLOCKSIZE>>>(scalar_d, scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(scalar_d, total_padded_32, ROW1);

    uint32_t* nonquote_scalar_d;
    cudaMalloc(&nonquote_scalar_d, total_padded_32*sizeof(uint32_t));
    start = clock();
    parallel_not<<<numBlock, BLOCKSIZE>>>(in_string_d, in_string_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(in_string_d, total_padded_32, ROW1);
    start = clock();
    parallel_and<uint32_t><<<numBlock, BLOCKSIZE>>>(scalar_d, in_string_d, nonquote_scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(nonquote_scalar_d, total_padded_32, ROW1);

    uint32_t* overflow;
    cudaMalloc(&overflow, total_padded_32*sizeof(uint32_t));
    start = clock();
    parallel_shift_right<<<numBlock, BLOCKSIZE>>>(nonquote_scalar_d, overflow, 31, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(overflow, total_padded_32, ROW1);

    uint32_t* follows_nonquote_scalar_d;
    cudaMalloc(&follows_nonquote_scalar_d, total_padded_32*sizeof(uint32_t));
    start = clock();
    parallel_shift_left<<<numBlock, BLOCKSIZE>>>(nonquote_scalar_d, follows_nonquote_scalar_d,  1, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(follows_nonquote_scalar_d, total_padded_32, ROW1);

    start = clock();
    parallel_or<<<numBlock, BLOCKSIZE>>>(follows_nonquote_scalar_d, overflow, follows_nonquote_scalar_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;


    //print_d(follows_nonquote_scalar_d, total_padded_32, ROW1);

    uint8_t* in_string_8_d;
    cudaMalloc(&in_string_8_d, size * sizeof(uint8_t));
    start = clock();
    split<<<numBlock, BLOCKSIZE>>>(in_string_d, in_string_8_d, size, total_padded_32);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print_d(in_string_d, total_padded_32, ROW1);
    
    //print8_d<int>(in_string_8_d, size, ROW1);

    uint8_t* in_string;
    in_string = (uint8_t* )malloc(sizeof(uint8_t)*size);
    start = clock();
    parallel_and<uint8_t><<<numBlock, BLOCKSIZE>>>(in_string_8_d, block_d, in_string_8_d, size, size);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    //print8_d<int>(in_string_8_d, size, ROW1);
    cudaMemcpy(in_string, in_string_8_d, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost);

    //print8(in_string, size, ROW1);

    return in_string;
}





long start(uint8_t * block, uint64_t size, long* res){
    uint8_t * block_d;
    uint64_t * parse_tree; 
    uint8_t* tokens;
    //printf("%c\n", (char)block[0]);
    cudaMalloc(&block_d, BUFSIZE*sizeof(uint8_t));
    cudaMemcpy(block_d, block, sizeof(uint8_t)*size, cudaMemcpyHostToDevice);

    /*bool isValidUTF8 = UTF8Validate(block_d, size);
    if(!isValidUTF8) {
        printf("not a valid utf input\n"); 
        exit(0);
    }*/
    clock_t start, end;
    start = clock();
    tokens = Tokenize(block_d, size);
    end = clock();
    double runtime = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //print8(tokens, size, ROW1);
    printf("total runtime: %f\n", runtime);
    return size-1;

} 





long *readFile(char* name){
    unsigned long  bytesread;
    uint8_t  buf[BUFSIZE];
    int   sizeLeftover=0;
    int   bLoopCompleted = 0;
    long  pos = 0;
    long * res;
    FILE * handle;
    // Open source file
    if (!(handle = fopen(name,"rb")))
    {
    // Bail
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
    // Process data - Replace with your function
    //
    // Function should return the position in the file or -1 if failed
    //
    // We are also passing bLoopCompleted to let ProcessData know whether this is
    // the last record (in which case - if no end-of-record separator,
    // use eof and process anyway)
        pos = start(buf, bytesread+sizeLeftover, res);
        pos = 0;
   
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
        sizeLeftover = std::min(bytesread+sizeLeftover-pos, sizeof(buf)-MAXLINELENGTH);
    // Extra protection - should never happen but you can never be too safe
        if (sizeLeftover<1) sizeLeftover=0;     
                     
    // If we have a leftover unprocessed buffer, move it to the beginning of 
    // read buffer so that when reading the next block, it will connect to the
    // current leftover and together complete a full readable line
        if (pos!=0 && sizeLeftover!=0)
        memmove(buf, buf+pos, sizeLeftover);
 
    } while(!bLoopCompleted);
    return res;
  // Close file
  fclose(handle); 
}


int main(int argc, char **argv)
{
  long* result;
  if (argv[1] != NULL){
    if( strcmp(argv[1], "-b") == 0 && argv[2] != NULL){
      std::cout << "Batch mode..." << std::endl;
      result = readFile(argv[2]);
    }
    else std::cout << "Command should be like '-b[file path]'" << std::endl;
  }
  else{
    std::cout << "Please select (batch: -b): " << std::endl;
  }
  return 0;
}