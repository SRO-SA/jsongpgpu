#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <x86intrin.h>
#include "cuda_profiler_api.h"
#include "DyckNew_Parallel_GPU.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <string.h>
#include "JSON_Parallel_Parser_Threads.h"

struct not_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x > 0);
    }
};

struct is_minus
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x < 0);
    }
};


const char * FILENAMES[]={"./inputs/Long_8.txt", "./inputs/Long_16.txt", "./inputs/Long_32.txt", "./inputs/Long_64.txt"};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

double  step1=0,
        step2=0,
        step3=0,
        step4=0,
        step5=0, 
        step6=0, 
        step7=0, 
        step8=0, 
        scanStep=0, 
        lastStep=0,
        correct1=0,
        correct2=0,
        correct3=0,
        correct4=0,
        program=0;

#define RUNTIMES 50

#define BLOCKSIZE 128
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

bool isCorrect(int strLength, int32_t* input, char* string);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int print(int32_t* input, int length, int rows){
  for(int32_t i =0; i<rows; i++){
    for(int32_t j=0; j<length; j++){
      std::cout << *(input+j+(i*length)) << ' ';
    }
    std::cout << std::endl;
  }
  return 1;
}

int printString(char* input, int length, int rows){
  for(int32_t i =0; i<rows; i++){
    for(int32_t j=0; j<length; j++){
      std::cout << *(input+j+(i*length)) << ' ';
    }
    std::cout << std::endl;
  }
  return 1;  
}

double runMultipleTimes(double(*func)()){
  double runtime = 0.0;

  for(int i=0; i<RUNTIMES; i++){
    runtime += func();
  }
  runtime = runtime/RUNTIMES;
  step1= step1/RUNTIMES;
  step2= step2/RUNTIMES;
  step3= step3/RUNTIMES;
  step4= step4/RUNTIMES;
  step5= step5/RUNTIMES; 
  step6= step6/RUNTIMES; 
  step7= step7/RUNTIMES; 
  step8= step8/RUNTIMES;
  scanStep= scanStep/RUNTIMES; 
  lastStep= lastStep/RUNTIMES;
  correct1= correct1/RUNTIMES;
  correct2= correct2/RUNTIMES;
  correct3= correct3/RUNTIMES;
  correct4= correct4/RUNTIMES;
  //program= program/RUNTIMES;
  std::cout << "First step mean time for " << RUNTIMES << " number of runs: " << step1 << "ms." << std::endl;
  std::cout << "Second step mean time for " << RUNTIMES << " number of runs: " << step2 << "ms." << std::endl;
  std::cout << "Correctenss First step mean time for " << RUNTIMES << " number of runs: " << correct1 << "ms." << std::endl;
  std::cout << "Correctenss Second step mean time for " << RUNTIMES << " number of runs: " << correct2 << "ms." << std::endl;
  std::cout << "Correctenss Third step mean time for " << RUNTIMES << " number of runs: " << correct3 << "ms." << std::endl;
  std::cout << "Correctenss Fourth step mean time for " << RUNTIMES << " number of runs: " << correct4 << "ms." << std::endl;
  std::cout << "Third step mean time for " << RUNTIMES << " number of runs: " << step3 << "ms." << std::endl;
  std::cout << "Fourth step mean time for " << RUNTIMES << " number of runs: " <<step4 << "ms." << std::endl;
  std::cout << "Fifth step mean time for " << RUNTIMES << " number of runs: " << step5 << "ms." << std::endl;
  std::cout << "Sixth step mean time for " << RUNTIMES << " number of runs: " << step6 << "ms." << std::endl;
  std::cout << "Seventh step mean time for " << RUNTIMES << " number of runs: " << step7 << "ms." << std::endl;
  std::cout << "Eighth step mean time for " << RUNTIMES << " number of runs: " << step8 << "ms." << std::endl;
  std::cout << "Scan step mean time for " << RUNTIMES << " number of runs: " << scanStep << "ms." << std::endl;
  std::cout << "Last step mean time for " << RUNTIMES << " number of runs: " << lastStep << "ms." << std::endl;

  std::cout << "Mean time for " << RUNTIMES << " number of runs: " << runtime << "ms." << std::endl;
  //std::cout << "Internal Mean time for " << RUNTIMES << " number of runs: " << program << "ms." << std::endl;

  step1=0;
  step2=0;
  step3=0;
  step4=0;
  step5=0; 
  step6=0; 
  step7=0; 
  step8=0;
  scanStep=0; 
  lastStep=0;
  correct1=0;
  correct2=0;
  correct3=0;
  correct4=0;
  return runtime;
}

__global__
void inv(int32_t length, int32_t * arr, int32_t * res){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    res[arr[ROW1*length + i]] = i;
  }
}

int32_t *sort(int length, int numBlock, int32_t * arr)
{
  clock_t start, end;
  double total = 0;
  //int32_t* cudaArr;
  start = clock();
  //cudaMallocAsync(&cudaArr, length*ROW2*sizeof(int32_t),0);
  cudaMemcpyAsync(arr + length*ROW2, arr, length*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  //thrust::device_ptr<int32_t> devArr(arr + length*ROW3);
  end = clock();

  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("allocation time: %f\n", total);

  start = clock();
  thrust::stable_sort_by_key(thrust::cuda::par, arr + length*ROW2, arr + length*ROW3, arr + length*ROW3);
  end = clock();

  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("sort time: %f\n", total);

  //int32_t *res;
  //cudaMallocAsync(&res, length*ROW1*sizeof(int32_t),0);
  start = clock();
  inv<<<numBlock, BLOCKSIZE>>>(length, arr + length*ROW2, arr + length*ROW4);
  cudaStreamSynchronize(0);
  end = clock();

  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("inverse time: %f\n", total);
  
  //cudaFreeAsync(cudaArr,0);
  return arr + length*ROW4;
}

__global__
void initialize(int step, int length, char* strArr, int32_t* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride){
    int currentChar = (int) strArr[i];
    if(currentChar == OPENBRACKET || currentChar == OPENBRACE || currentChar == COMMA){
      res[i] = 1;
    }
    if(currentChar == OPENBRACKET || currentChar == OPENBRACE){
      res[length + i] = 1;
    }
    else if(currentChar == CLOSEBRACKET || currentChar == CLOSEBRACE){
      res[length + i] = -1;
      res[ROW3*length + i] = 1;
    }
    if(currentChar == OPENBRACE || currentChar == OPENBRACKET || currentChar == CLOSEBRACE || currentChar == CLOSEBRACKET){
      res[ROW2*length + i] = 1;
    }
    if(currentChar != CLOSEBRACE && currentChar != CLOSEBRACKET){
      res[ROW4*length + i] = i;
    }
  }
}

__global__
void add_depth(char* string, int32_t *arr1, int32_t* arr2, int length){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    if(string[i] == COMMA) arr1[i] = arr1[i] + arr2[i-1];
  }
}

__global__
void changeDepth(int length, char* strArr, char* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    int currentChar = (int) strArr[i];
    if(currentChar == COMMA){
      while(i>0 && (strArr[i-1] == CLOSEBRACKET || strArr[i-1] == CLOSEBRACE)){
        *(res+i) = *(strArr+i-1);
        *(res+i-1) = (char)currentChar;
        i--;
      }
      *(res+i) = currentChar;
      if(strArr[i-1] == OPENBRACKET || strArr[i-1] == OPENBRACE){
        *(res+i) = (char)I;
      }
    }
    else if(res[i] == 0){
      *(res+i) = *(strArr+i);
    }
  }
}

int32_t findDepthAndCount(int length, int numBlock, int32_t** arr, char * string)
{
  //cudaMallocAsync(arr, length*ROW3*sizeof(int32_t),0);
  //cudaEvent_t gpu_start, gpu_stop;
  float runtime = 0;
  //cudaEventCreate(&gpu_start);
  //cudaEventCreate(&gpu_stop);

  //cudaEventRecord(gpu_start);
  cudaMemset(*arr, 0, ROW5*length*sizeof(int32_t));
  initialize<<<numBlock, BLOCKSIZE>>>(0, length, string, *arr);
  cudaStreamSynchronize(0);
  //cudaEventRecord(gpu_stop);
  //cudaEventSynchronize(gpu_stop);
  //gpuErrchk( cudaPeekAtLastError() );
  thrust::inclusive_scan(thrust::cuda::par, (*arr), (*arr) + length, (*arr));
  thrust::inclusive_scan(thrust::cuda::par, (*arr) + length, (*arr) + ROW2*length, (*arr) + length);
  thrust::exclusive_scan(thrust::cuda::par, (*arr) + ROW2*length, (*arr) + ROW3*length, (*arr) + ROW2*length);
  thrust::inclusive_scan_by_key(thrust::cuda::par, (*arr) + ROW4*length, (*arr) + ROW5*length, (*arr) + ROW3*length, (*arr) + ROW4*length);



  //cudaEventElapsedTime(&runtime, gpu_start, gpu_stop);

  // std::cout << "-------------Second Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << runtime << std::endl;
  // // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW1);
  // // cudaMemcpyAsync(h_long_test, (*arr)+length, sizeof(int32_t)*length*ROW1, cudaMemcpyDeviceToHost);
  // // print(h_long_test, length, ROW1);
  // // free(h_long_test);
  // std::cout << "-------------End Second Step--------------" << std::endl;

  add_depth<<<numBlock, BLOCKSIZE>>>(string, (*arr) + length, (*arr) + ROW3*length, length);

  int32_t res;
  cudaMemcpyAsync(&res, (*arr)+(ROW2*length)-2, sizeof(int32_t), cudaMemcpyDeviceToHost);
  if(res == 0){
    return 1;
  }
  return -1;
}

__global__
void extract(int length, int arrLength, char* string, int32_t* arr, char* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    int currentChar = (int)string[i];
    if(i<length && (currentChar == OPENBRACE || currentChar == OPENBRACKET || currentChar == CLOSEBRACE || currentChar == CLOSEBRACKET)){
      res[arr[i]] = string[i];
    }
  }
}

//INPUT Currectness Check BEGIN
__global__
void countNodesRepititionStep(int length, int32_t* arr, int i, int32_t* res){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t j=index; j<length; j+=stride){
    if(i == -1){
      res[j] = arr[j];
      if(j>0 && arr[j]==arr[j-1]){
        res[length + j] = 1;
      }
      else{
        res[length + j] = 0;
      }
    }
    if(i > -1){
      int pow2 = 1<<i;
      res[j] = arr[j];
      if(j >= pow2){
        if((arr[j] == arr[j - pow2])){
          res[length + j] = arr[length + j - pow2] + arr[length + j];
        }
        else {
          res[length + j] = arr[length + j];
        }
      }
      else{
        res[length + j] = arr[length + j];
      }
    }
  }  
}

int32_t* countNodesRepitition(int length, int numBlock, int32_t* arr)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __builtin_clz(length-1));
  int32_t * cudaArr;
  int32_t * cudaRes;
  cudaMallocAsync(&cudaArr, length*ROW2*sizeof(int32_t),0);
  cudaMallocAsync(&cudaRes, length*ROW2*sizeof(int32_t),0);
  cudaMemcpyAsync(cudaArr, arr,  length*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  int i = -1;

  for(int n = nextP2*2; n>1; n=n>>1){
    countNodesRepititionStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, cudaRes);
    cudaStreamSynchronize(0);
    cudaMemcpyAsync(cudaArr, cudaRes,  length*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);    
    i+=1;
  }
  cudaFreeAsync(cudaArr,0);
  return (cudaRes);
}

 __global__
void checkCurrectenss(int length, char* string, int32_t* arr, int32_t* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i=index; i<length; i+=stride){
    int currentChar = (int) string[i];
    if(currentChar == CLOSEBRACE || currentChar == CLOSEBRACKET){
      int32_t value = arr[i] - 1;
      int32_t base = i - arr[i];
      while(value>0){
        base--;
        if(arr[base]==0){
          --value;
        }
        else{
          value = value + arr[base] - 1;
          base = base - arr[base];
        }
      }
      int openning = (int) string[base];
      if((currentChar == CLOSEBRACE && openning == OPENBRACE)||(currentChar == CLOSEBRACKET && openning == OPENBRACKET)){
        res[i] = 1;
      }
      else{
        res[i] = 0;
      }
    }
    else{
      res[i] = 1;
    }
  }
}

__global__
void set_open_odd_close_even(char* input_d, int32_t* o_o_c_e, int32_t* o_e_c_o, int length){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i=index; i<length; i+=stride){
    char currentChar = input_d[i];
    o_o_c_e[i] = ((i & 1) == ((currentChar & 2) >> 1)); // odd opening and even closing (== works as XNOR)
    o_e_c_o[i] = ((i & 1) ^  ((currentChar & 2) >> 1)); // even opening and odd closing
  }
}

__global__
void check_is_matched(char* input_d, uint8_t* res_check, int length){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i=index; i<length; i+=stride){
    char currentChar = input_d[i*2];
    char nextChar = input_d[i*2+1];
      uint8_t sixth_bit_i = (currentChar >> 5) & 1;
      uint8_t second_bit_i = (currentChar >> 1) & 1;
      uint8_t sixth_bit_i_1 = (nextChar >> 5) & 1;
      uint8_t second_bit_i_1 = (nextChar >> 1) & 1; 

      res_check[i] = (sixth_bit_i == sixth_bit_i_1) && (second_bit_i == 1) && (second_bit_i ^ second_bit_i_1) ? 0 : 1;
      //matched &= (sixth_bit_i == sixth_bit_i_1) && (second_bit_i == 1) && (second_bit_i ^ second_bit_i_1);    

  }
}


bool matching(char *input_d, int length, int iter, int numBlock){
  bool matched = true;
  if(length < 3 || iter == 1){
    uint8_t* res_check;
    int length_divided = length/2;
    int dividedNumBlock = ((length_divided) + BLOCKSIZE - 1) / BLOCKSIZE;

    cudaMallocAsync(&res_check, sizeof(uint8_t)*length_divided,0);
    check_is_matched<<<dividedNumBlock, BLOCKSIZE>>>(input_d, res_check, length_divided);
    matched = thrust::reduce(thrust::cuda::par, res_check, res_check+length_divided) == 0 ? true : false;
    cudaFreeAsync(res_check,0);
    // char * res_check = (char *)malloc(sizeof(char)*length);
    // cudaMemcpyAsync(res_check, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
    // for(int i = 0; i< length; i+=2){
    //   uint8_t sixth_bit_i = (res_check[i] >> 5) & 1;
    //   uint8_t second_bit_i = (res_check[i] >> 1) & 1;
    //   uint8_t sixth_bit_i_1 = (res_check[i+1] >> 5) & 1;
    //   uint8_t second_bit_i_1 = (res_check[i+1] >> 1) & 1; 

    //   matched &= (sixth_bit_i == sixth_bit_i_1) && (second_bit_i == 1) && (second_bit_i ^ second_bit_i_1);    

    // }
    // free(res_check);

    // if(!matched){
    //   std::cout << "-------------Curretness "<< iter << " Step--------------" << std::endl;
    //   //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    //   char * h_char_test = (char*) malloc(sizeof(char)*length);
    //   cudaMemcpyAsync(h_char_test, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
    //   printString(h_char_test, length, ROW1);
    //   free(h_char_test);
    //   std::cout << "-------------End "<< iter <<" Step--------------" << std::endl;
    
    // }
    return matched;
  }
  int32_t * o_o_c_e;

  int32_t * o_e_c_o;


  cudaMallocAsync(&o_o_c_e, sizeof(int32_t)*length,0);
  cudaMallocAsync(&o_e_c_o, sizeof(int32_t)*length,0);


  set_open_odd_close_even<<<numBlock, BLOCKSIZE>>>(input_d, o_o_c_e, o_e_c_o, length);
  cudaStreamSynchronize(0);

  char * right_reduced;
  char * left_reduced;
  int32_t right_length;
  int32_t left_length;

  left_length = thrust::count_if(thrust::cuda::par, o_o_c_e, o_o_c_e+length, not_zero());
  right_length = thrust::count_if(thrust::cuda::par, o_e_c_o, o_e_c_o+length, not_zero());
  // printf("left: %d\n", left_length);
  // printf("right: %d\n", right_length);

  if(left_length == 0 || right_length == 0){
    uint8_t* res_check;
    int length_divided = length/2;
    int dividedNumBlock = ((length_divided) + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaMallocAsync(&res_check, sizeof(uint8_t)*length_divided,0);
    check_is_matched<<<dividedNumBlock, BLOCKSIZE>>>(input_d, res_check, length_divided);
    matched = thrust::reduce(thrust::cuda::par, res_check, res_check+length_divided) == 0 ? true : false;
    cudaFreeAsync(res_check,0);
    cudaFreeAsync(o_e_c_o,0);
    cudaFreeAsync(o_o_c_e,0);

    // char * res_check = (char *)malloc(sizeof(char)*length);
    // cudaMemcpyAsync(res_check, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
    // for(int i = 0; i< length; i+=2){
    //   uint8_t sixth_bit_i = (res_check[i] >> 5) & 1;
    //   uint8_t second_bit_i = (res_check[i] >> 1) & 1;
    //   uint8_t sixth_bit_i_1 = (res_check[i+1] >> 5) & 1;
    //   uint8_t second_bit_i_1 = (res_check[i+1] >> 1) & 1; 

    //   matched &= (sixth_bit_i == sixth_bit_i_1) && (second_bit_i == 1) && (second_bit_i ^ second_bit_i_1);

    // }
    // free(res_check);
    // if(!matched){
    //   std::cout << "-------------Curretness "<< iter << " Step--------------" << std::endl;
    //   //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
    //   char * h_char_test = (char*) malloc(sizeof(char)*length);
    //   cudaMemcpyAsync(h_char_test, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
    //   printString(h_char_test, length, ROW1);
    //   free(h_char_test);
    //   std::cout << "-------------End "<< iter <<" Step--------------" << std::endl;
    
    // }
    return matched;
  }

  //cudaMemcpyAsync(&right_length, o_e_c_o+length-1, sizeof(int32_t), cudaMemcpyDeviceToHost);
  //cudaMemcpyAsync(&left_length, o_o_c_e+length-1, sizeof(int32_t), cudaMemcpyDeviceToHost);



  cudaMallocAsync(&right_reduced, sizeof(char)*right_length,0);
  cudaMallocAsync(&left_reduced, sizeof(char)*left_length,0);

  thrust::copy_if(thrust::cuda::par, input_d, input_d+length, o_e_c_o, right_reduced, not_zero());
  thrust::copy_if(thrust::cuda::par, input_d, input_d+length, o_o_c_e, left_reduced, not_zero());
  bool right_res;
  bool left_res;

  cudaFreeAsync(o_e_c_o,0);
  cudaFreeAsync(o_o_c_e,0);

  right_res = matching(right_reduced, right_length, iter>>1, ((length) + BLOCKSIZE - 1) / BLOCKSIZE);
  left_res = matching(left_reduced, left_length, iter >> 1,  ((length) + BLOCKSIZE - 1) / BLOCKSIZE);

  cudaFreeAsync(right_reduced,0);
  cudaFreeAsync(left_reduced,0);

  return (right_res && left_res);
}

bool isCorrect(int strLength, int32_t* input, char* string)
{
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  int32_t* h_long_test;
  allStart = clock();
  int arrLength;
  cudaMemcpyAsync(&arrLength, input + strLength - 2, sizeof(int32_t), cudaMemcpyDeviceToHost);
  arrLength++;
  int numBlock = ((arrLength) + BLOCKSIZE - 1) / BLOCKSIZE;
  char* res;

  start = clock();
  cudaMallocAsync(&res, arrLength*sizeof(char),0);
  extract<<<numBlock, BLOCKSIZE>>>(strLength, arrLength, string, input, res);
  cudaStreamSynchronize(0);
  end = clock();
  correct1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Curretness First Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // printf("%d\n", arrLength);
  //h_char_test = (char*) malloc(sizeof(char)*arrLength);
  //cudaMemcpyAsync(h_char_test, res, sizeof(char)*arrLength, cudaMemcpyDeviceToHost);
  //printString(h_char_test, arrLength, ROW1);
  //free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

  int nextP2 = arrLength == 1 ? 1 : 1 << (32 - __builtin_clz(arrLength-1));

  start = clock();
  bool isCorrect =  matching(res, arrLength, nextP2, numBlock);
  end = clock();

  allEnd = clock();
  cudaFreeAsync(res,0);
  // std::cout << "-------------isCorrect--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000 << std::endl;
  // printf("%d\n", isCorrect);
  // std::cout << "-------------End isCorrect--------------" << std::endl;

  return isCorrect;
  
  /*
  int32_t* arr;
  start = clock();
  cudaMallocAsync(&arr, arrLength*sizeof(int32_t),0);
  initialize<<<numBlock, BLOCKSIZE>>>(1, arrLength, res, arr);
  cudaStreamSynchronize(0);
  //gpuErrchk( cudaPeekAtLastError() );

  int32_t* longRes;
  thrust::inclusive_scan(thrust::cuda::par, arr, arr + arrLength, arr);
  end = clock();
  correct2 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  std::cout << "-------------Curretness Second Step--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength);
  // cudaMemcpyAsync(h_long_test, arr, sizeof(int32_t)*arrLength, cudaMemcpyDeviceToHost);
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  std::cout << "-------------End Second Step--------------" << std::endl;

  start = clock();
  longRes = countNodesRepitition(arrLength, numBlock, arr);
  end = clock();
  correct3 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  std::cout << "-------------Curretness Third Step--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (int32_t *)malloc(sizeof(int32_t)*arrLength);
  // cudaMemcpyAsync(h_long_test, longRes, sizeof(int32_t)*arrLength, cudaMemcpyDeviceToHost); 
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  std::cout << "-------------End Third Step--------------" << std::endl;

  start = clock();
  checkCurrectenss<<<numBlock, BLOCKSIZE>>>(arrLength, res, (longRes+arrLength), arr);
  cudaStreamSynchronize(0);
  thrust::inclusive_scan(thrust::cuda::par, arr, arr + arrLength, arr);
  end = clock();
  correct4 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  std::cout << "-------------Curretness Fourth Step--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (int32_t *)malloc(sizeof(int32_t)*arrLength);
  // cudaMemcpyAsync(h_long_test, arr, sizeof(int32_t)*arrLength, cudaMemcpyDeviceToHost); 
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  std::cout << "-------------End Fourth Step--------------" << std::endl;

  allEnd = clock();

  cudaFreeAsync(res,0);
  cudaFreeAsync(longRes,0);
  int32_t isCorrect;
  cudaMemcpyAsync(&isCorrect, arr+arrLength-1, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaFreeAsync(arr,0);
  std::cout << "-------------isCorrect--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000 << std::endl;
  // printf("%d\n", isCorrect == arrLength);
  std::cout << "-------------End isCorrect--------------" << std::endl;
  return isCorrect == arrLength;
  */
}
//INPUT Currectness Check END

__global__
void reduce(int length, int arrLength, char * string, int32_t * arr, int32_t * res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    int currentChar = (int)string[i];
    if(i < length && (currentChar == OPENBRACKET || currentChar == OPENBRACE || currentChar == COMMA)){
      if(i != 0) {
        res[arr[i-1]] = arr[length + i - 1];
        res[arrLength + arr[i-1]] = arr[i-1];
      }
      else{
        res[0] = 0;
        res[arrLength] = 0;
      } 
    }
  }
}

int32_t * sortByDepth(int length, int numBlock, int32_t * arr)
{
  //int32_t * res;
  int32_t* tmp;
  //cudaMallocAsync(&res, length*ROW2*sizeof(int32_t),0);
  //cudaMemcpyAsync(res, arr,  length*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  tmp = sort(length, numBlock, arr);
  cudaMemcpyAsync((arr+length*ROW3), tmp, length*ROW1*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  //cudaFreeAsync(tmp,0);
  return arr+length*ROW2;
}

__global__
void findParents(int length, int32_t * arr, int32_t * res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i=index; i<length-1; i+=stride){
    if(arr[i+1] == arr[i] + 1){
      res[arr[length + i + 1]] = arr[length + i];
    }
  }
}

__global__
void propagateParentsAndCountChildrenStep(int length, int32_t* arr, int32_t* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t j=index; j<length; j+=stride){
    res[j] = arr[j];
    res[length+j] = j != 0 ? 1 : -1; 
    // if(j != 0) res[length + j] = 1;
    // else res[length + j] = -1;
  }  
}

__global__
void propagateParents(int length, int32_t* arr, int32_t* keys, int32_t* res){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i=index; i<length; i+=stride){
    res[i] = (keys[i] & i < 0 ) ? arr[arr[i]] : arr[i];
  } 

}

int32_t* propagateParentsAndCountChildren(int length, int numBlock, int32_t* arr, int32_t* res)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __builtin_clz(length-1));
  clock_t start, end;
  //int32_t * cudaArr;
  //int32_t * keys_d;
  //int32_t * res;
  //int32_t * index_d;
  thrust::plus<int> op;
  int32_t first_index = -1;

  // int32_t * h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, arr, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  //cudaArr = arr + length*ROW1 -> arr + length*ROW3 - 1
  //keys_d  = arr + length*ROW3 -> arr + length*ROW4 - 1
  //index_d = arr + length*ROW4 -> arr + length*ROW5 - 1

  //cudaMallocAsync(&cudaArr, length*ROW2*sizeof(int32_t),0);
  //cudaMallocAsync(&keys_d, length*ROW1*sizeof(int32_t),0);
  //cudaMallocAsync(&index_d, length*ROW1*sizeof(int32_t),0);

  //cudaMallocAsync(&res, length*ROW2*sizeof(int32_t),0);
  cudaMemcpyAsync(arr + length*ROW1, arr,  length*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(arr + length*ROW3, arr,  length*sizeof(int32_t), cudaMemcpyDeviceToDevice);

  int i = -1;
  propagateParentsAndCountChildrenStep<<<numBlock, BLOCKSIZE>>>(length, arr + length*ROW1, arr + length*ROW1);  
  cudaStreamSynchronize(0);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, cudaArr, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  thrust::inclusive_scan_by_key(thrust::cuda::par, arr + length*ROW3, arr + length*ROW4, arr + length*ROW1, arr + length*ROW1);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, cudaArr, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);
  
  cudaMemcpyAsync(arr + length*ROW3, arr + length*ROW1,  length*sizeof(int32_t), cudaMemcpyDeviceToDevice);

  thrust::sequence(thrust::cuda::par, arr + length*ROW4, arr + length*ROW5);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW1);
  // cudaMemcpyAsync(h_long_test, index_d, sizeof(int32_t)*length*ROW1, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW1);
  // free(h_long_test);

  thrust::transform_if(thrust::cuda::par, arr + length*ROW1, arr + length*ROW2, arr + length*ROW4, arr + length*ROW3, arr + length*ROW1, op, is_minus());
  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, cudaArr, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  // *(res+i) = *(cudaArr+*(cudaArr+i)) if key+i < 0
  propagateParents<<<numBlock, BLOCKSIZE>>>(length, arr + length*ROW1, arr + length*ROW3, arr + length*ROW1);    
  cudaStreamSynchronize(0);
  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, cudaArr, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  //cudaMemcpyAsync(cudaArr, &first_index, sizeof(int32_t), cudaMemcpyHostToDevice);
  //cudaMemcpyAsync(cudaArr+length, &first_index, sizeof(int32_t), cudaMemcpyHostToDevice);

  //cudaMemcpyAsync(keys_d, cudaArr,  length*sizeof(int32_t), cudaMemcpyDeviceToDevice);

  thrust::inclusive_scan_by_key(thrust::cuda::par, arr + length*ROW1, arr + length*ROW2, arr + length*ROW2, arr + length*ROW2);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, cudaArr, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  cudaMemcpyAsync(res, arr + length*ROW1, length*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);

  //double max = 0;
  //double total = 0;
  /*for(int n = nextP2*2; n>1; n=n>>1){
    //start = clock();
    propagateParentsAndCountChildrenStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, res);    
    cudaStreamSynchronize(0);
    //end = clock();
    //double time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //max = max > time ? max : time;
    //total += time;
    cudaMemcpyAsync(cudaArr, res,  length*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
    i+=1;
  }*/
  //printf("parents max time: %f, %d\n", max, nextP2);
  //printf("parents total time: %f\n", total);

  //cudaFreeAsync(cudaArr,0);
  //cudaFreeAsync(index_d,0);
  //cudaFreeAsync(keys_d,0);

  return res;
}

__global__
void childsNumber(int length, int32_t* arr, int32_t* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i=index; i<length; i+=stride){
    res[i] = arr[i];
    res[length + i] = arr[length + i];
    res[length*ROW2 + i] = 0;
    if(i == length - 1) res[length*ROW2 + arr[i]] = arr[length + i];
    else if(arr[i] != arr[i + 1] && arr[i] != -1) res[length*ROW2 + arr[i]] = arr[length + i];
    if(i == 0 && length == 1) res[length*ROW2] = 0;
  }
}

__global__
void addOne(int length, int32_t* arr)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if(index==0) arr[length*ROW3] = 0;
  for(int32_t i=index; i<length; i+=stride){
    arr[length*ROW3 + i] = arr[length*ROW3 + i] + 1;
  }
  if(index==0) arr[length*ROW3] = 0;
}

int32_t * allocate(int length, int numBlock, int32_t* arr, int32_t* res)
{
  //int32_t * cudaArr;
  //cudaMallocAsync(&cudaArr, length*ROW4*sizeof(int32_t),0);
  cudaMemcpyAsync(res, arr,  length*ROW3*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(res+length*ROW3+1, arr+length*ROW2,  (length*ROW1-1)*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  addOne<<<numBlock, BLOCKSIZE>>>(length, res);
  cudaStreamSynchronize(0);
  thrust::inclusive_scan(thrust::cuda::par, res+ROW3*length, res + ROW4*length, res+ROW3*length);
  return res;
}

int32_t scan(int length, int32_t* arr)
{
  //int32_t * cudaArr;
  //int32_t * res;
  //cudaMallocAsync(&cudaArr, length*ROW4*sizeof(int32_t),0);
  //cudaMallocAsync(&res, length*ROW1*sizeof(int32_t),0);
  //cudaMemcpyAsync(cudaArr, arr,  length*ROW4*sizeof(int32_t), cudaMemcpyHostToDevice); 
  int32_t res = thrust::reduce(thrust::cuda::par, arr+ROW2*length, arr + ROW3*length);
  //cudaFreeAsync(cudaArr,0);
  return res;
}

__global__
void generateRes(int length, int32_t* arr, int32_t* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  
  for(int32_t i = index; i<length; i+=stride){
    res[arr[length*ROW3 + i]] = arr[length*ROW2 + i];
    if(arr[i] != -1) res[arr[length*ROW3+ arr[i]]+arr[length+ i]] = arr[length*ROW3+ i];
  }
}


double NewRuntime_Parallel_GPU(char* input_d, int length) {
  //*******************************//
  // size_t l_free = 0;
  // size_t l_Total = 0;
  // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
  // size_t allocated = (l_Total - l_free);
  // std::cout << "Total: " << l_Total << " Free: " << l_free << " Allocated: " << allocated << std::endl;
  //*******************************//
  //exit(0);
  //cudaProfilerStart();
  int attachedLength = length;
  int numBlock = (attachedLength + BLOCKSIZE - 1) / BLOCKSIZE;
  int32_t* res;
  int32_t* fakeRes;
  int32_t* arr;
  cudaMallocAsync(&arr, attachedLength*ROW5*sizeof(int32_t),0);
  //printf("length: %d\n", attachedLength);
  //char* attacheArr;
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  int32_t* h_long_test;
  allStart = clock();

  start = clock();

  //attacheArr = input;
  //memcpy(attacheArr, input, length*sizeof(char));
  //attacheArr[length] = ',';
  char* d_attacheArr;
  cudaMallocAsync(&d_attacheArr, attachedLength*sizeof(char),0);
  cudaMemcpyAsync(d_attacheArr, input_d, attachedLength*sizeof(char), cudaMemcpyDeviceToDevice);
  
  //cudaMallocAsyncManaged(&attacheArr, attachedLength*sizeof(char),0);
  //cudaMemcpyAsync(attacheArr, input, length*sizeof(char), cudaMemcpyHostToDevice);
  //attacheArr[length] = ',';
  //char* d_sameDepthArr;
  //cudaMallocAsync(&d_sameDepthArr, attachedLength*sizeof(char),0);
  //cudaMemcpyAsync(d_sameDepthArr, input_d, attachedLength*sizeof(char), cudaMemcpyDeviceToDevice);

  //changeDepth<<<numBlock, BLOCKSIZE>>>(attachedLength, d_attacheArr, d_sameDepthArr);
  //cudaStreamSynchronize(0);
  //free(attacheArr);
  //cudaFreeAsync(d_attacheArr,0);
  //end = clock();
  //step1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------First Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // // h_char_test = (char*)malloc(sizeof(char)*attachedLength);
  // // cudaMemcpyAsync(h_char_test, d_sameDepthArr, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
  // // printString(h_char_test, attachedLength, ROW1);
  // // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

    //*******************************//
    // size_t l_free = 0;
    // size_t l_Total = 0;
    // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
    // size_t allocated = (l_Total - l_free);
    // std::cout << "Total: " << l_Total << " Free: " << l_free << " Allocated: " << allocated << std::endl;
    //*******************************//
  start = clock();
  int32_t *d_arr;
  int32_t correctDepth;
  correctDepth = findDepthAndCount(attachedLength, numBlock, &arr, d_attacheArr);
  end = clock();
  step2 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Second Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // // h_long_test = (int32_t*)malloc(sizeof(int32_t)*attachedLength*ROW3);
  // // cudaMemcpyAsync(h_long_test, arr, sizeof(int32_t)*attachedLength*ROW3, cudaMemcpyDeviceToHost);
  // // print(h_long_test, attachedLength, ROW3);
  // // free(h_long_test);
  // std::cout << "-------------End Second Step--------------" << std::endl;
  int32_t arrLength;
  cudaMemcpyAsync(&arrLength, arr+(attachedLength-1), sizeof(int32_t), cudaMemcpyDeviceToHost);
  //cudaMallocAsync(&res, arrLength*ROW4*sizeof(int32_t),0);


  /*
      1. count open bracket brace and comma for node count. /inclusive Row 1
      2. count both open and close bracket and braces for depth. /inclusive Row 2
      3. count open and close bracket braces for corretness check. /exclusive Row 3
      4. count close bracket and braces. /inclusive Row 4
      5. subtract 4 from depth to find correct depth for all outside comma.
      6. reduce input for correctness and reduce it for parsing.
      7. call isCorrect.
      8. continue to sortByDepth...
  */
  //exit(0);

  if(correctDepth != -1){
    bool correct;
    correct = isCorrect(attachedLength, arr+(attachedLength)*ROW2, d_attacheArr);
    if(correct){
      start = clock();
      //printf("         %d\n", attachedLength);

      //cudaMallocAsync(&arr, attachedLength*ROW4*sizeof(int32_t),0);
      //cudaMemcpyAsync(arr, d_arr,  attachedLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      //cudaFreeAsync(d_arr,0);

      cudaMallocAsync(&res, arrLength*ROW4*sizeof(int32_t),0);
      reduce<<<numBlock, BLOCKSIZE>>>(attachedLength, arrLength, d_attacheArr, arr, res);
      cudaStreamSynchronize(0);
      end = clock();
      step3 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Third Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW2);
      // // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW2);
      // // free(h_long_test);
      // std::cout << "-------------End Third Step--------------" << std::endl;
      //cudaFreeAsync(d_sameDepthArr,0);
      int numBlock = (arrLength + BLOCKSIZE - 1) / BLOCKSIZE;

      start = clock();
      cudaMemcpyAsync(arr, res,  arrLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      fakeRes = sortByDepth(arrLength, numBlock, arr);
      end = clock();
      step4 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Fourth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (int32_t *)malloc(sizeof(int32_t)*arrLength*ROW2);
      // // cudaMemcpyAsync(h_long_test, fakeRes, sizeof(int32_t)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW2);
      // // free(h_long_test);
      // std::cout << "-------------End Fourth Step--------------" << std::endl;

      start = clock();
      cudaMemcpyAsync(arr, fakeRes,  arrLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      //cudaFreeAsync(fakeRes,0);
      cudaMemset(res, -1, arrLength*ROW1*sizeof(int32_t));
      findParents<<<numBlock, BLOCKSIZE>>>( arrLength, arr, res);
      cudaStreamSynchronize(0);
      end = clock();
      step5 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Fifth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW1);
      // // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW1);
      // // free(h_long_test);
      // std::cout << "-------------End Fifth Step--------------" << std::endl;


      start = clock();
      cudaMemcpyAsync(arr, res,  arrLength*ROW1*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      fakeRes = propagateParentsAndCountChildren(arrLength, numBlock, arr, res);
      end = clock();
      step6 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Sixth Step--------------" << std::endl;      
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // //  h_long_test = (int32_t*)malloc(sizeof(int32_t)*arrLength*ROW2);
      // //  cudaMemcpyAsync(h_long_test, fakeRes, sizeof(int32_t)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // //  print(h_long_test, arrLength, ROW2);
      // //  free(h_long_test);
      // std::cout << "-------------End Sixth Step--------------" << std::endl;
      start = clock();
      cudaMemcpyAsync(arr, fakeRes,  arrLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      //cudaFreeAsync(fakeRes,0);
      cudaMemset(res, -1, arrLength*ROW3*sizeof(int32_t));
      childsNumber<<<numBlock, BLOCKSIZE>>>(arrLength, arr, res);
      cudaStreamSynchronize(0);
      end = clock();
      step7 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Seventh Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW3);
      // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*arrLength*ROW3, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW3);
      // free(h_long_test);
      // std::cout << "-------------End Seventh Step--------------" << std::endl;
      
      start = clock();
      cudaMemcpyAsync(arr, res,  arrLength*ROW3*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      fakeRes = allocate(arrLength, numBlock, arr, res);
      end = clock();
      step8 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Eighth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW4);
      // cudaMemcpyAsync(h_long_test, fakeRes, sizeof(int32_t)*arrLength*ROW4, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW4);
      // free(h_long_test);
      // std::cout << "-------------End Eighth Step--------------" << std::endl;

      //int32_t* sumRes;
      int32_t resLength;
      //cudaMallocAsync(&sumRes, arrLength*ROW1*sizeof(int32_t),0);
      start = clock();
      //thrust::reduce(thrust::cuda::par, fakeRes+ROW2*arrLength, fakeRes+ROW3*arrLength);
      resLength = scan(arrLength, fakeRes);
      end = clock();
      scanStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;

      // std::cout << "-------------Scan Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW1);
      // // cudaMemcpyAsync(h_long_test, sumRes, sizeof(int32_t)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW1);
      // // free(h_long_test);
      // std::cout << "-------------End Scan Step--------------" << std::endl;
      //cudaMemcpyAsync(&resLength, sumRes + arrLength - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
      start = clock();
      cudaMemcpyAsync(arr, fakeRes,  arrLength*ROW4*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      //cudaFreeAsync(sumRes,0);
      //cudaFreeAsync(fakeRes,0);
      cudaFreeAsync(res,0);
      cudaMallocAsync(&res, (arrLength+resLength)*sizeof(int32_t),0);
      cudaMemset(res, 0, (arrLength+resLength)*sizeof(int32_t));
      generateRes<<<numBlock, BLOCKSIZE>>>(arrLength,  arr, res);
      cudaStreamSynchronize(0);
      end = clock();
      lastStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Last Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // //h_long_test = (int32_t*) malloc(sizeof(int32_t)*(arrLength+resLength)*ROW1);
      // //cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*(arrLength+resLength)*ROW1, cudaMemcpyDeviceToHost);
      // //print(h_long_test, (arrLength+resLength), ROW1);
      // //free(h_long_test);
      // std::cout << "-------------End Last Step--------------" << std::endl;

      cudaFreeAsync(arr,0);
      cudaFreeAsync(res,0);
      allEnd = clock();
      //cudaProfilerStop();    
      //*******************************//
      // size_t l_free = 0;
      // size_t l_Total = 0;
      // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
      // size_t allocated = (l_Total - l_free);
      // std::cout << "Total: " << l_Total << " Free: " << l_free << " Allocated: " << allocated << std::endl;
      //*******************************//
      //program += ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000;
      //printf("program: %f\n", program);

    } 
    else{
      printf("Input wrong\n");
      return 0;  
    }
  }
  else {
    printf("Input invalid\n");
    return 0;
  }
  return (double)(allEnd-allStart);
}
