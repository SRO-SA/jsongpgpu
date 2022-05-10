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
#include "JSON_Parallel_Parser.h"

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

bool isCorrect(int strLength, long* input, char* string);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int print(long* input, int length, int rows){
  for(long i =0; i<rows; i++){
    for(long j=0; j<length; j++){
      std::cout << *(input+j+(i*length)) << ' ';
    }
    std::cout << std::endl;
  }
  return 1;
}

int printString(char* input, int length, int rows){
  for(long i =0; i<rows; i++){
    for(long j=0; j<length; j++){
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
void inv(long length, long * arr, long * res){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i = index; i< length; i+=stride)
  {
    res[arr[ROW1*length + i]] = i;
  }
}

long *sort(int length, int numBlock, long * arr)
{
  clock_t start, end;
  double total = 0;
  long* cudaArr;
  start = clock();
  cudaMalloc(&cudaArr, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr, length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
  thrust::device_ptr<long> devArr(cudaArr);
  end = clock();

  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("allocation time: %f\n", total);

  start = clock();
  thrust::stable_sort_by_key(thrust::cuda::par, cudaArr, cudaArr+length, cudaArr+length);
  end = clock();

  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("sort time: %f\n", total);

  long *res;
  cudaMalloc(&res, length*ROW1*sizeof(long));
  start = clock();
  inv<<<numBlock, BLOCKSIZE>>>(length, cudaArr, res);
  cudaDeviceSynchronize();
  end = clock();

  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("inverse time: %f\n", total);
  
  cudaFree(cudaArr);
  return res;
}

__global__
void initialize(int step, int length, char* strArr, long* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i = index; i< length; i+=stride){
    int currentChar = (int) strArr[i];
    if(step == 0){
      if(currentChar == OPENBRACKET || currentChar == OPENBRACE || currentChar == COMMA){
        res[i] = 1;
      }
      else{
        res[i] = 0;
      }
      if(currentChar == OPENBRACKET || currentChar == OPENBRACE){
        res[length + i] = 1;
      }
      else if(currentChar == CLOSEBRACKET || currentChar == CLOSEBRACE){
        res[length + i] = -1;
      }
      else {
        res[length + i] = 0;        
      }
      if(currentChar == OPENBRACE || currentChar == OPENBRACKET || currentChar == CLOSEBRACE || currentChar == CLOSEBRACKET){
        res[ROW2*length + i] = 1;
      }
      else{
        res[ROW2*length + i] = 0;
      }
    }
    if(step == 1){
      if(currentChar == OPENBRACE || currentChar == OPENBRACKET){
        res[i] = 1;
      }
      else{ 
        res[i] = 0;
      }
    }
  }
}

__global__
void changeDepth(int length, char* strArr, char* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i = index; i< length; i+=stride)
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

long findDepthAndCount(int length, int numBlock, long** arr, char * string)
{
  cudaMalloc(arr, length*ROW3*sizeof(long));
  initialize<<<numBlock, BLOCKSIZE>>>(0, length, string, *arr);
  cudaDeviceSynchronize();
  //gpuErrchk( cudaPeekAtLastError() );
  thrust::inclusive_scan(thrust::cuda::par, (*arr), (*arr) + length, (*arr));
  thrust::inclusive_scan(thrust::cuda::par, (*arr) + length, (*arr) + ROW2*length, (*arr) + length);
  thrust::exclusive_scan(thrust::cuda::par, (*arr) + ROW2*length, (*arr) + ROW3*length, (*arr) + ROW2*length);
  long res;
  cudaMemcpy(&res, (*arr)+(ROW2*length)-1, sizeof(long), cudaMemcpyDeviceToHost);
  if(res == 0){
    return 1;
  }
  return -1;
}

__global__
void extract(int length, int arrLength, char* string, long* arr, char* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i = index; i< length; i+=stride)
  {
    int currentChar = (int)string[i];
    if(i<length && (currentChar == OPENBRACE || currentChar == OPENBRACKET || currentChar == CLOSEBRACE || currentChar == CLOSEBRACKET)){
      res[arr[i]] = string[i];
    }
  }
}

//INPUT Currectness Check BEGIN
__global__
void countNodesRepititionStep(int length, long* arr, int i, long* res){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long j=index; j<length; j+=stride){
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

long* countNodesRepitition(int length, int numBlock, long* arr)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __builtin_clz(length-1));
  long * cudaArr;
  long * cudaRes;
  cudaMalloc(&cudaArr, length*ROW2*sizeof(long));
  cudaMalloc(&cudaRes, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*sizeof(long), cudaMemcpyDeviceToDevice);
  int i = -1;

  for(int n = nextP2*2; n>1; n=n>>1){
    countNodesRepititionStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, cudaRes);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaArr, cudaRes,  length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);    
    i+=1;
  }
  cudaFree(cudaArr);
  return (cudaRes);
}

 __global__
void checkCurrectenss(int length, char* string, long* arr, long* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i=index; i<length; i+=stride){
    int currentChar = (int) string[i];
    if(currentChar == CLOSEBRACE || currentChar == CLOSEBRACKET){
      long value = arr[i] - 1;
      long base = i - arr[i];
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
void set_open_odd_close_even(char* input_d, uint32_t* o_o_c_e, uint32_t* o_e_c_o, int length){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i=index; i<length; i+=stride){
    char currentChar = input_d[i];
    o_o_c_e[i] = ((i & 1) == ((currentChar & 2) >> 1)); // odd opening and even closing (== works as XNOR)
    o_e_c_o[i] = ((i & 1) ^  ((currentChar & 2) >> 1)); // even opening and odd closing
  }
}

__global__
void check_is_matched(char* input_d, uint8_t* res_check, int length){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i=index; i<length; i+=stride){
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

    cudaMalloc(&res_check, sizeof(uint8_t)*length_divided);
    check_is_matched<<<dividedNumBlock, BLOCKSIZE>>>(input_d, res_check, length_divided);
    matched = thrust::reduce(thrust::cuda::par, res_check, res_check+length_divided) == 0 ? true : false;
    cudaFree(res_check);
    // char * res_check = (char *)malloc(sizeof(char)*length);
    // cudaMemcpy(res_check, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
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
    //   cudaMemcpy(h_char_test, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
    //   printString(h_char_test, length, ROW1);
    //   free(h_char_test);
    //   std::cout << "-------------End "<< iter <<" Step--------------" << std::endl;
    
    // }
    return matched;
  }
  uint32_t * o_o_c_e;

  uint32_t * o_e_c_o;


  cudaMalloc(&o_o_c_e, sizeof(uint32_t)*length);
  cudaMalloc(&o_e_c_o, sizeof(uint32_t)*length);


  set_open_odd_close_even<<<numBlock, BLOCKSIZE>>>(input_d, o_o_c_e, o_e_c_o, length);
  cudaDeviceSynchronize();

  char * right_reduced;
  char * left_reduced;
  uint32_t right_length;
  uint32_t left_length;

  left_length = thrust::count_if(thrust::cuda::par, o_o_c_e, o_o_c_e+length, not_zero());
  right_length = thrust::count_if(thrust::cuda::par, o_e_c_o, o_e_c_o+length, not_zero());
  // printf("left: %d\n", left_length);
  // printf("right: %d\n", right_length);

  if(left_length == 0 || right_length == 0){
    uint8_t* res_check;
    int length_divided = length/2;
    int dividedNumBlock = ((length_divided) + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaMalloc(&res_check, sizeof(uint8_t)*length_divided);
    check_is_matched<<<dividedNumBlock, BLOCKSIZE>>>(input_d, res_check, length_divided);
    matched = thrust::reduce(thrust::cuda::par, res_check, res_check+length_divided) == 0 ? true : false;
    cudaFree(res_check);
    cudaFree(o_e_c_o);
    cudaFree(o_o_c_e);

    // char * res_check = (char *)malloc(sizeof(char)*length);
    // cudaMemcpy(res_check, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
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
    //   cudaMemcpy(h_char_test, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
    //   printString(h_char_test, length, ROW1);
    //   free(h_char_test);
    //   std::cout << "-------------End "<< iter <<" Step--------------" << std::endl;
    
    // }
    return matched;
  }

  //cudaMemcpy(&right_length, o_e_c_o+length-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  //cudaMemcpy(&left_length, o_o_c_e+length-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);



  cudaMalloc(&right_reduced, sizeof(char)*right_length);
  cudaMalloc(&left_reduced, sizeof(char)*left_length);

  thrust::copy_if(thrust::cuda::par, input_d, input_d+length, o_e_c_o, right_reduced, not_zero());
  thrust::copy_if(thrust::cuda::par, input_d, input_d+length, o_o_c_e, left_reduced, not_zero());
  bool right_res;
  bool left_res;

  cudaFree(o_e_c_o);
  cudaFree(o_o_c_e);

  right_res = matching(right_reduced, right_length, iter>>1, ((length) + BLOCKSIZE - 1) / BLOCKSIZE);
  left_res = matching(left_reduced, left_length, iter >> 1,  ((length) + BLOCKSIZE - 1) / BLOCKSIZE);

  cudaFree(right_reduced);
  cudaFree(left_reduced);

  return (right_res && left_res);
}

bool isCorrect(int strLength, long* input, char* string)
{
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  long* h_long_test;
  allStart = clock();
  int arrLength;
  cudaMemcpy(&arrLength, input + strLength - 1, sizeof(long), cudaMemcpyDeviceToHost);
  arrLength++;
  int numBlock = ((arrLength) + BLOCKSIZE - 1) / BLOCKSIZE;
  char* res;

  start = clock();
  cudaMalloc(&res, arrLength*sizeof(char));
  extract<<<numBlock, BLOCKSIZE>>>(strLength, arrLength, string, input, res);
  cudaDeviceSynchronize();
  end = clock();
  correct1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Curretness First Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // printf("%d\n", arrLength);
  // //h_char_test = (char*) malloc(sizeof(char)*arrLength);
  // //cudaMemcpy(h_char_test, res, sizeof(char)*arrLength, cudaMemcpyDeviceToHost);
  // //printString(h_char_test, arrLength, ROW1);
  // //free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

  int nextP2 = arrLength == 1 ? 1 : 1 << (32 - __builtin_clz(arrLength-1));

  start = clock();
  bool isCorrect =  matching(res, arrLength, nextP2, numBlock);
  end = clock();

  allEnd = clock();
  cudaFree(res);
  // std::cout << "-------------isCorrect--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000 << std::endl;
  // printf("%d\n", isCorrect);
  // std::cout << "-------------End isCorrect--------------" << std::endl;

  return isCorrect;
  
  /*
  long* arr;
  start = clock();
  cudaMalloc(&arr, arrLength*sizeof(long));
  initialize<<<numBlock, BLOCKSIZE>>>(1, arrLength, res, arr);
  cudaDeviceSynchronize();
  //gpuErrchk( cudaPeekAtLastError() );

  long* longRes;
  thrust::inclusive_scan(thrust::cuda::par, arr, arr + arrLength, arr);
  end = clock();
  correct2 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  std::cout << "-------------Curretness Second Step--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (long*) malloc(sizeof(long)*arrLength);
  // cudaMemcpy(h_long_test, arr, sizeof(long)*arrLength, cudaMemcpyDeviceToHost);
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  std::cout << "-------------End Second Step--------------" << std::endl;

  start = clock();
  longRes = countNodesRepitition(arrLength, numBlock, arr);
  end = clock();
  correct3 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  std::cout << "-------------Curretness Third Step--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (long *)malloc(sizeof(long)*arrLength);
  // cudaMemcpy(h_long_test, longRes, sizeof(long)*arrLength, cudaMemcpyDeviceToHost); 
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  std::cout << "-------------End Third Step--------------" << std::endl;

  start = clock();
  checkCurrectenss<<<numBlock, BLOCKSIZE>>>(arrLength, res, (longRes+arrLength), arr);
  cudaDeviceSynchronize();
  thrust::inclusive_scan(thrust::cuda::par, arr, arr + arrLength, arr);
  end = clock();
  correct4 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  std::cout << "-------------Curretness Fourth Step--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (long *)malloc(sizeof(long)*arrLength);
  // cudaMemcpy(h_long_test, arr, sizeof(long)*arrLength, cudaMemcpyDeviceToHost); 
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  std::cout << "-------------End Fourth Step--------------" << std::endl;

  allEnd = clock();

  cudaFree(res);
  cudaFree(longRes);
  long isCorrect;
  cudaMemcpy(&isCorrect, arr+arrLength-1, sizeof(long), cudaMemcpyDeviceToHost);
  cudaFree(arr);
  std::cout << "-------------isCorrect--------------" << std::endl;
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000 << std::endl;
  // printf("%d\n", isCorrect == arrLength);
  std::cout << "-------------End isCorrect--------------" << std::endl;
  return isCorrect == arrLength;
  */
}
//INPUT Currectness Check END

__global__
void reduce(int length, int arrLength, char * string, long * arr, long * res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i = index; i< length; i+=stride)
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

long * sortByDepth(int length, int numBlock, long * arr)
{
  long * res;
  long* tmp;
  cudaMalloc(&res, length*ROW2*sizeof(long));
  cudaMemcpy(res, arr,  length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
  tmp = sort(length, numBlock, arr);
  cudaMemcpy((res+length), tmp, length*ROW1*sizeof(long), cudaMemcpyDeviceToDevice);
  cudaFree(tmp);
  return res;
}

__global__
void findParents(int length, long * arr, long * res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i=index; i<length-1; i+=stride){
    if(arr[i+1] == arr[i] + 1){
      res[arr[length + i + 1]] = arr[length + i];
    }
  }
}

__global__
void propagateParentsAndCountChildrenStep(int length, long* arr, long* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long j=index; j<length; j+=stride){
    res[j] = arr[j];
    if(j != 0) res[length + j] = 1;
    else res[length + j] = -1;
  }  
}

__global__
void propagateParents(int length, long* arr, long* keys, long* res){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i=index; i<length; i+=stride){
    res[i] = (keys[i] < 0 && i>0) ? arr[arr[i]] : arr[i];
  } 

}

long* propagateParentsAndCountChildren(int length, int numBlock, long* arr)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __builtin_clz(length-1));
  //clock_t start, end;
  long * cudaArr;
  long * keys_d;
  long * res;
  long * index_d;
  thrust::plus<int> op;
  long first_index = -1;

  // long * h_long_test = (long*)malloc(sizeof(long)*length*ROW2);
  // cudaMemcpy(h_long_test, arr, sizeof(long)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  cudaMalloc(&cudaArr, length*ROW2*sizeof(long));
  cudaMalloc(&keys_d, length*ROW1*sizeof(long));
  cudaMalloc(&index_d, length*ROW1*sizeof(long));

  cudaMalloc(&res, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*sizeof(long), cudaMemcpyDeviceToDevice);
  cudaMemcpy(keys_d, arr,  length*sizeof(long), cudaMemcpyDeviceToDevice);

  int i = -1;
  propagateParentsAndCountChildrenStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, cudaArr);  
  cudaDeviceSynchronize();

  // long* h_long_test = (long*)malloc(sizeof(long)*length*ROW2);
  // cudaMemcpy(h_long_test, cudaArr, sizeof(long)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  thrust::inclusive_scan_by_key(thrust::cuda::par, keys_d, keys_d+length, cudaArr, cudaArr);

  // long* h_long_test = (long*)malloc(sizeof(long)*length*ROW2);
  // cudaMemcpy(h_long_test, cudaArr, sizeof(long)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);
  
  cudaMemcpy(keys_d, cudaArr,  length*sizeof(long), cudaMemcpyDeviceToDevice);
  thrust::sequence(thrust::cuda::par, index_d, index_d+length);

  thrust::transform_if(thrust::cuda::par, cudaArr, cudaArr+length, index_d, keys_d, cudaArr, op, is_minus());
  // long* h_long_test = (long*)malloc(sizeof(long)*length*ROW2);
  // cudaMemcpy(h_long_test, cudaArr, sizeof(long)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  // *(res+i) = *(cudaArr+*(cudaArr+i)) if key+i < 0
  propagateParents<<<numBlock, BLOCKSIZE>>>(length, cudaArr, keys_d, cudaArr);    
  cudaDeviceSynchronize();
  // long* h_long_test = (long*)malloc(sizeof(long)*length*ROW2);
  // cudaMemcpy(h_long_test, cudaArr, sizeof(long)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  //cudaMemcpy(cudaArr, &first_index, sizeof(long), cudaMemcpyHostToDevice);
  //cudaMemcpy(cudaArr+length, &first_index, sizeof(long), cudaMemcpyHostToDevice);

  //cudaMemcpy(keys_d, cudaArr,  length*sizeof(long), cudaMemcpyDeviceToDevice);

  thrust::inclusive_scan_by_key(thrust::cuda::par, cudaArr, cudaArr+length, cudaArr+length, cudaArr+length);

  // long* h_long_test = (long*)malloc(sizeof(long)*length*ROW2);
  // cudaMemcpy(h_long_test, cudaArr, sizeof(long)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  cudaMemcpy(res, cudaArr, length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);

  //double max = 0;
  //double total = 0;
  /*for(int n = nextP2*2; n>1; n=n>>1){
    //start = clock();
    propagateParentsAndCountChildrenStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, res);    
    cudaDeviceSynchronize();
    //end = clock();
    //double time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    //max = max > time ? max : time;
    //total += time;
    cudaMemcpy(cudaArr, res,  length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
    i+=1;
  }*/
  //printf("parents max time: %f, %d\n", max, nextP2);
  //printf("parents total time: %f\n", total);

  cudaFree(cudaArr);
  cudaFree(index_d);
  cudaFree(keys_d);

  return res;
}

__global__
void childsNumber(int length, long* arr, long* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i=index; i<length; i+=stride){
    res[i] = arr[i];
    res[length + i] = arr[length + i];
    res[length*ROW2 + i] = 0;
    if(i == length - 1) res[length*ROW2 + arr[i]] = arr[length + i];
    else if(arr[i] != arr[i + 1] && arr[i] != -1) res[length*ROW2 + arr[i]] = arr[length + i];
    if(i == 0 && length == 1) res[length*ROW2] = 0;
  }
}

__global__
void addOne(int length, long* arr)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if(index==0) arr[length*ROW3] = 0;
  for(long i=index; i<length; i+=stride){
    arr[length*ROW3 + i] = arr[length*ROW3 + i] + 1;
  }
  if(index==0) arr[length*ROW3] = 0;
}

long * allocate(int length, int numBlock, long* arr)
{
  long * cudaArr;
  cudaMalloc(&cudaArr, length*ROW4*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*ROW3*sizeof(long), cudaMemcpyDeviceToDevice);
  cudaMemcpy(cudaArr+length*ROW3+1, arr+length*ROW2,  (length*ROW1-1)*sizeof(long), cudaMemcpyDeviceToDevice);
  addOne<<<numBlock, BLOCKSIZE>>>(length, cudaArr);
  cudaDeviceSynchronize();
  thrust::inclusive_scan(thrust::cuda::par, cudaArr+ROW3*length, cudaArr + ROW4*length, cudaArr+ROW3*length);
  return cudaArr;
}

long * scan(int length, long* arr)
{
  long * cudaArr;
  long * res;
  cudaMalloc(&cudaArr, length*ROW4*sizeof(long));
  cudaMalloc(&res, length*ROW1*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*ROW4*sizeof(long), cudaMemcpyHostToDevice); 
  thrust::inclusive_scan(thrust::cuda::par, cudaArr+ROW2*length, cudaArr + ROW3*length, res);
  cudaFree(cudaArr);
  return res;  
}

__global__
void generateRes(int length, long* arr, long* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  
  for(long i = index; i<length; i+=stride){
    res[arr[length*ROW3 + i]] = arr[length*ROW2 + i];
    if(arr[i] != -1) res[arr[length*ROW3+ arr[i]]+arr[length+ i]] = arr[length*ROW3+ i];
  }
}


double NewRuntime_Parallel_GPU(char* input_d, int length) {
  //cudaProfilerStart();
  int attachedLength = length;
  int numBlock = (attachedLength + BLOCKSIZE - 1) / BLOCKSIZE;
  long* res;
  long* fakeRes;
  long* arr;
  //char* attacheArr;
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  long* h_long_test;
  allStart = clock();

  start = clock();

  //attacheArr = input;
  //memcpy(attacheArr, input, length*sizeof(char));
  //attacheArr[length] = ',';
  char* d_attacheArr;
  cudaMalloc(&d_attacheArr, attachedLength*sizeof(char));
  cudaMemcpy(d_attacheArr, input_d, attachedLength*sizeof(char), cudaMemcpyDeviceToDevice);
  
  //cudaMallocManaged(&attacheArr, attachedLength*sizeof(char));
  //cudaMemcpy(attacheArr, input, length*sizeof(char), cudaMemcpyHostToDevice);
  //attacheArr[length] = ',';
  char* d_sameDepthArr;
  cudaMalloc(&d_sameDepthArr, attachedLength*sizeof(char));
  cudaMemcpy(d_sameDepthArr, input_d, attachedLength*sizeof(char), cudaMemcpyDeviceToDevice);

  changeDepth<<<numBlock, BLOCKSIZE>>>(attachedLength, d_attacheArr, d_sameDepthArr);
  cudaDeviceSynchronize();
  //free(attacheArr);
  cudaFree(d_attacheArr);
  end = clock();
  step1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------First Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // // h_char_test = (char*)malloc(sizeof(char)*attachedLength);
  // // cudaMemcpy(h_char_test, d_sameDepthArr, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
  // // printString(h_char_test, attachedLength, ROW1);
  // // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

  start = clock();
  long *d_arr;
  long correctDepth;
  correctDepth = findDepthAndCount(attachedLength, numBlock, &d_arr, d_sameDepthArr);
  end = clock();
  step2 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Second Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // // h_long_test = (long*)malloc(sizeof(long)*attachedLength*ROW3);
  // // cudaMemcpy(h_long_test, d_arr, sizeof(long)*attachedLength*ROW3, cudaMemcpyDeviceToHost);
  // // print(h_long_test, attachedLength, ROW3);
  // // free(h_long_test);
  // std::cout << "-------------End Second Step--------------" << std::endl;
  long arrLength;
  cudaMemcpy(&arrLength, d_arr+(attachedLength-1), sizeof(long), cudaMemcpyDeviceToHost);
  if(correctDepth != -1){
    bool correct;
    correct = isCorrect(attachedLength, d_arr+(attachedLength)*ROW2, d_sameDepthArr);
    if(correct){      
      start = clock();
      cudaMalloc(&arr, attachedLength*ROW4*sizeof(long));
      cudaMalloc(&res, arrLength*ROW4*sizeof(long));
      cudaMemcpy(arr, d_arr,  attachedLength*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
      reduce<<<numBlock, BLOCKSIZE>>>(attachedLength, arrLength, d_sameDepthArr, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      step3 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Third Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW2);
      // // cudaMemcpy(h_long_test, res, sizeof(long)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW2);
      // // free(h_long_test);
      // std::cout << "-------------End Third Step--------------" << std::endl;
      cudaFree(d_sameDepthArr);
      cudaFree(d_arr);
      int numBlock = (arrLength + BLOCKSIZE - 1) / BLOCKSIZE;

      start = clock();
      cudaMemcpy(arr, res,  arrLength*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
      fakeRes = sortByDepth(arrLength, numBlock, arr);
      end = clock();
      step4 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Fourth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (long *)malloc(sizeof(long)*arrLength*ROW2);
      // // cudaMemcpy(h_long_test, fakeRes, sizeof(long)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW2);
      // // free(h_long_test);
      // std::cout << "-------------End Fourth Step--------------" << std::endl;

      start = clock();
      cudaMemcpy(arr, fakeRes,  arrLength*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
      cudaFree(fakeRes);
      cudaMemset(res, -1, arrLength*ROW1*sizeof(long));
      findParents<<<numBlock, BLOCKSIZE>>>( arrLength, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      step5 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Fifth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW1);
      // // cudaMemcpy(h_long_test, res, sizeof(long)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW1);
      // // free(h_long_test);
      // std::cout << "-------------End Fifth Step--------------" << std::endl;
      
      start = clock();
      cudaMemcpy(arr, res,  arrLength*ROW1*sizeof(long), cudaMemcpyDeviceToDevice);
      fakeRes = propagateParentsAndCountChildren(arrLength, numBlock, arr);
      end = clock();
      step6 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Sixth Step--------------" << std::endl;      
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // //  h_long_test = (long*)malloc(sizeof(long)*arrLength*ROW2);
      // //  cudaMemcpy(h_long_test, fakeRes, sizeof(long)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // //  print(h_long_test, arrLength, ROW2);
      // //  free(h_long_test);
      // std::cout << "-------------End Sixth Step--------------" << std::endl;

      start = clock();
      cudaMemcpy(arr, fakeRes,  arrLength*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
      cudaFree(fakeRes);
      cudaMemset(res, -1, arrLength*ROW3*sizeof(long));
      childsNumber<<<numBlock, BLOCKSIZE>>>(arrLength, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      step7 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Seventh Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW3);
      // // cudaMemcpy(h_long_test, res, sizeof(long)*arrLength*ROW3, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW3);
      // // free(h_long_test);
      // std::cout << "-------------End Seventh Step--------------" << std::endl;
      
      start = clock();
      cudaMemcpy(arr, res,  arrLength*ROW3*sizeof(long), cudaMemcpyDeviceToDevice);
      fakeRes = allocate(arrLength, numBlock, arr);
      end = clock();
      step8 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Eighth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW4);
      // // cudaMemcpy(h_long_test, fakeRes, sizeof(long)*arrLength*ROW4, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW4);
      // // free(h_long_test);
      // std::cout << "-------------End Eighth Step--------------" << std::endl;

      long* sumRes;
      //cudaMalloc(&sumRes, arrLength*ROW1*sizeof(long));
      start = clock();
      sumRes = scan(arrLength, fakeRes);
      end = clock();
      scanStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Scan Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW1);
      // // cudaMemcpy(h_long_test, sumRes, sizeof(long)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW1);
      // // free(h_long_test);
      // std::cout << "-------------End Scan Step--------------" << std::endl;
      long resLength;
      cudaMemcpy(&resLength, sumRes + arrLength - 1, sizeof(long), cudaMemcpyDeviceToHost);
      start = clock();
      cudaMemcpy(arr, fakeRes,  arrLength*ROW4*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(sumRes);
      cudaFree(fakeRes);
      cudaFree(res);
      cudaMalloc(&res, (arrLength+resLength)*sizeof(long));
      cudaMemset(res, 0, (arrLength+resLength)*sizeof(long));
      generateRes<<<numBlock, BLOCKSIZE>>>(arrLength,  arr, res);
      cudaDeviceSynchronize();
      end = clock();
      lastStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Last Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // //h_long_test = (long*) malloc(sizeof(long)*(arrLength+resLength)*ROW1);
      // //cudaMemcpy(h_long_test, res, sizeof(long)*(arrLength+resLength)*ROW1, cudaMemcpyDeviceToHost);
      // //print(h_long_test, (arrLength+resLength), ROW1);
      // //free(h_long_test);
      // std::cout << "-------------End Last Step--------------" << std::endl;

      cudaFree(arr);
      cudaFree(res);
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
