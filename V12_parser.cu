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

#include <sys/resource.h>
#include <stdint.h>

uint64_t time_used ( ) {
   struct rusage ru;
   struct timeval t;
   getrusage(RUSAGE_THREAD,&ru);
   t = ru.ru_utime;
   return (uint64_t) t.tv_sec*1000 + t.tv_usec/1000;
}

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

struct decrease
{
  __host__ __device__
  int operator()(int x)
  {
    return x-1;
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


struct is_opening
{
  __host__ __device__
  bool operator()(char x)
  {
    return (x==OPENBRACE) || (x==OPENBRACKET);
  }
};

struct is_not_bracket_brace
{
  __host__ __device__
  bool operator()(const char c)
  {
    return  !((c==OPENBRACE) || (c==OPENBRACKET) || (c==CLOSEBRACE) || (c==CLOSEBRACKET));
  }

};

struct is_not_open_comma
{
  __host__ __device__
  bool operator()(const char c)
  {
    return  !((c==OPENBRACE) || (c==OPENBRACKET) || (c==COMMA));
  }

};



inline bool isCorrect(int strLength, int32_t* input, char* string);

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
    for(int32_t j=0; j < 600; j++){ //232470
      if (j == 503) printf("la In ");
      std::cout << *(input+j+(i*length)) << ' ';
      //if(*(input+j+(i*length)) == 0)  std::cout << "R: " << i << " I: " << j << " value: " << *(input+j+(i*length)) << ' ';
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

__global__
void inv(int32_t length, int32_t * arr, int32_t * res, int32_t* index_arr, int32_t* index_res){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    res[arr[ROW1*length + i]] = i;
    //index_res[arr[ROW1*length + i]] = index_arr[i];
  }
}

inline int32_t *sort(int length, int numBlock, int32_t * arr, int32_t* input_index_d)
{
  clock_t start, end;
  double total = 0;
  //int32_t* cudaArr;
  start = clock();
  //cudaMallocAsync(&cudaArr, length*ROW2*sizeof(int32_t),0);
  // R1 Depth
  // R2 Node
  // R3 Depth
  // R4 Node
  // R5 Depth
  cudaMemcpyAsync(arr + length*ROW2, arr, length*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(arr + length*ROW4, arr, length*ROW1*sizeof(int32_t), cudaMemcpyDeviceToDevice);

  //thrust::device_ptr<int32_t> devArr(arr + length*ROW3);
  end = clock();

  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("allocation time: %f\n", total);

  start = clock();
  thrust::stable_sort_by_key(thrust::cuda::par, arr + length*ROW2, arr + length*ROW3, arr + length*ROW3);
  end = clock();
  
  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW4);
  // cudaMemcpyAsync(h_long_test, arr + length*ROW2, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost, 0);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);
  // printf("______________________________________\n");

  thrust::stable_sort_by_key(thrust::cuda::par, arr + length*ROW4, arr + length*ROW5, input_index_d);

  // h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW4);
  // cudaMemcpyAsync(h_long_test, input_index_d, sizeof(int32_t)*length*ROW1, cudaMemcpyDeviceToHost, 0);
  // print(h_long_test, length, ROW1);
  // free(h_long_test);
  // printf("______________________________________\n");
  // //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("sort time: %f\n", total);

  //int32_t *res;
  //cudaMallocAsync(&res, length*ROW1*sizeof(int32_t),0);
  //start = clock();
  //inv<<<numBlock, BLOCKSIZE>>>(length, arr + length*ROW2, arr + length*ROW4, input_index_d, arr+length*ROW5);
  //cudaStreamSynchronize(0);
  //end = clock();
  //cudaMemcpyAsync(input_index_d, arr+length*ROW5, sizeof(int32_t)*length*ROW1, cudaMemcpyDeviceToDevice, 0);
  // int32_t* h_long_test = (int32_t *)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, input_index_d, sizeof(int32_t)*length*ROW1, cudaMemcpyDeviceToHost, 0);
  // print(h_long_test, length, ROW1);
  // free(h_long_test);
  // printf("______________________________________\n");
  //total = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //printf("inverse time: %f\n", total);
  
  //cudaFreeAsync(cudaArr,0);
  return arr + length*ROW3;
}

__global__ 
void check_with_next(char* strArr, int32_t* res, int length){

  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride){
    uint8_t currentChar = strArr[(i<<1)];
    uint8_t nextChar = strArr[(i<<1)+1];
    uint8_t byte_high_current = (currentChar >> 4) & 0x0F;
    uint8_t byte_low_current = (currentChar) & 0x0F;
    uint8_t byte_high_next = (nextChar >> 4) & 0x0F;
    uint8_t byte_low_next = (nextChar) & 0x0F;

    res[i] = !((byte_high_current == byte_high_next) & (byte_low_current != byte_low_next) & (byte_low_current == 0x0B));
    //res[i] = ((strArr[i] == OPENBRACE && strArr[i+1] != CLOSEBRACE) || (strArr[i] == OPENBRACKET && strArr[i+1] != CLOSEBRACKET)) ? 1 : 0;
  }

}

__global__ 
void initialize_depth(char * strArr, int32_t* res, int length){
  constexpr int8_t lowbyte2_g[16] = {      // OPEN _B
    0,0,0,0,
    0,0,0,0,
    0,0,0,1,
    0,0,0,0
  };
  constexpr int8_t lowbyte3_g[16] = {      // CLOSE _D
    0,0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,1,0,0
  };
  constexpr int8_t highbyte1_g[16] = {     // OPEN CLOSE COMMA 5_ 7_ 2_
    0,0,1,0,
    0,1,0,1,
    0,0,0,0,
    0,0,0,0
  };

  int tid = threadIdx.x;
  __shared__ int8_t lowbyte2[16];
  __shared__ int8_t lowbyte3[16];
  __shared__ int8_t highbyte1[16];
  if(tid == 0){
    for(int k=0; k<16; k++){
      lowbyte2[k]=lowbyte2_g[k];
      lowbyte3[k]=lowbyte3_g[k];
      highbyte1[k]=highbyte1_g[k];
    }

  }
  __syncthreads();
  int index = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride){
    uint8_t currentChar = (uint8_t) strArr[i];
    uint8_t byte_high = (currentChar >> 4) & 0x0F;
    uint8_t byte_low = currentChar & 0x0F;
    res[i] = (lowbyte2[byte_low]&highbyte1[byte_high]) | -(lowbyte3[byte_low]&highbyte1[byte_high]);
  }
}

__global__ 
void initialize(int step, int length, char* strArr, int32_t* res)
{
  constexpr int8_t lowbyte1_g[16] = {       // OPEN  COMMA _B _C
    0,0,0,0,
    0,0,0,0,
    0,0,0,1,
    1,0,0,0
  };
  constexpr int8_t lowbyte2_g[16] = {      // OPEN _B
    0,0,0,0,
    0,0,0,0,
    0,0,0,1,
    0,0,0,0
  };
  constexpr int8_t lowbyte3_g[16] = {      // CLOSE _D
    0,0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,1,0,0
  };
  constexpr int8_t highbyte1_g[16] = {     // OPEN CLOSE COMMA 5_ 7_ 2_
    0,0,1,0,
    0,1,0,1,
    0,0,0,0,
    0,0,0,0
  };

  int tid = threadIdx.x;
  __shared__ int8_t lowbyte1[16];
  __shared__ int8_t lowbyte2[16];
  __shared__ int8_t lowbyte3[16];
  __shared__ int8_t highbyte1[16];
  if(tid == 0){
    for(int k=0; k<16; k++){
      lowbyte1[k]=lowbyte1_g[k];
      lowbyte2[k]=lowbyte2_g[k];
      lowbyte3[k]=lowbyte3_g[k];
      highbyte1[k]=highbyte1_g[k];
    }

  }
  __syncthreads();

  int index = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride){
    uint8_t currentChar = (uint8_t) strArr[i];
    uint8_t byte_high = (currentChar >> 4) & 0x0F;
    uint8_t byte_low = currentChar & 0x0F;
    int8_t highbyte1_res = highbyte1[byte_high];
    int8_t is_open = lowbyte2[byte_low]&highbyte1_res;
    int8_t is_close = lowbyte3[byte_low]&highbyte1_res;
    
    res[i] = lowbyte1[byte_low]&highbyte1_res;
    res[length+i] = is_open | -is_close;
    res[ROW2*length+i] = is_open | is_close;
    res[ROW3*length+i] = is_close;
    res[ROW4*length+i] = is_close;

  }
}

__global__ 
void add_depth(char* string, int32_t *arr1, int32_t* arr2, int length){ // Unused
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    char previous = i > 0 ? string[i-1] : 0;
    // if(string[i] == COMMA && (previous == CLOSEBRACE || previous == CLOSEBRACKET) &&  i==8366952)
    // printf("1: depth: %d, close before: %d, char: %c, previous: %c\n", arr1[i], arr2[i-1], string[i], previous); 
    if(string[i] == COMMA) arr1[i] = arr1[i] + arr2[i-1];
    // if(string[i] == COMMA && (previous == CLOSEBRACE || previous == CLOSEBRACKET) &&  i==8366952)
    // printf("2: depth: %d, close before: %d, char: %c, previous: %c\n", arr1[i], arr2[i-1], string[i], previous); 

  }
}

inline int32_t findDepthAndCount(int length, int numBlock, int32_t** arr, char * string)
{
  //cudaMallocAsync(arr, length*ROW3*sizeof(int32_t),0);
  cudaEvent_t gpu_start, gpu_stop;
  float runtime = 0;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  // cudaEventRecord(gpu_start);

  //cudaMemset(*arr, 0, ROW3*length*sizeof(int32_t));
  initialize<<<numBlock, BLOCKSIZE>>>(0, length, string, *arr);
  cudaStreamSynchronize(0);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW4);
  // cudaMemcpyAsync(h_long_test, (*arr)+length, sizeof(int32_t)*length*ROW4, cudaMemcpyDeviceToHost);
  // print(h_long_test+length*ROW1, length, ROW1);
  // free(h_long_test);

  // cudaEventRecord(gpu_stop);
  // cudaEventSynchronize(gpu_stop);
  //gpuErrchk( cudaPeekAtLastError() );

  thrust::inclusive_scan(thrust::cuda::par, (*arr), (*arr) + length, (*arr)); //ROW1: node count
  thrust::inclusive_scan(thrust::cuda::par, (*arr) + length, (*arr) + ROW2*length, (*arr) + length); //ROW2: depth count
  thrust::exclusive_scan(thrust::cuda::par, (*arr) + ROW2*length, (*arr) + ROW3*length, (*arr) + ROW2*length); //ROW3: Open 1 Close 1 (total Brace/bracket)
  thrust::inclusive_scan_by_key(thrust::cuda::par, (*arr) + ROW4*length, (*arr) + ROW5*length, (*arr) + ROW3*length, (*arr) + ROW3*length); // Consecutive closing



  // cudaeventElapsedTime(&runtime, gpu_start, gpu_stop);

  // std::cout << "-------------Second////// Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << runtime << std::endl;
  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW4);
  // cudaMemcpyAsync(h_long_test, (*arr), sizeof(int32_t)*length*ROW4, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW4);
  // free(h_long_test);
  // std::cout << "-------------End Second/////// Step--------------" << std::endl;

  add_depth<<<numBlock, BLOCKSIZE>>>(string, (*arr) + length, (*arr) + ROW3*length, length);
  cudaStreamSynchronize(0);
  

  // h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW4);
  // cudaMemcpyAsync(h_long_test, (*arr), sizeof(int32_t)*length*ROW4, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW4);
  // free(h_long_test);
  // std::cout << "-------------End Second/////// Step--------------" << std::endl;
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

inline bool isCorrect(int strLength, int32_t* input, char* string)
{
  cudaEvent_t gpu_start, gpu_stop;
  float elapsed_time = 0;
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  int32_t* h_long_test;
  allStart = clock();
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  int arrLength;
  cudaMemcpyAsync(&arrLength, input + strLength - 2, sizeof(int32_t), cudaMemcpyDeviceToHost);
  arrLength++;
  int numBlock = ((arrLength) + BLOCKSIZE - 1) / BLOCKSIZE;
  char* res;
  int32_t * instance;

  // cudaEventRecord(gpu_start);


  start = clock();
  cudaMallocAsync(&res, arrLength*sizeof(char),0);
  thrust::remove_copy_if(thrust::cuda::par, string, string+strLength, res, is_not_bracket_brace());
  // extract<<<numBlock, BLOCKSIZE>>>(strLength, arrLength, string, input, res);
  // cudaStreamSynchronize(0);
  end = clock();
  correct1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Curretness First Step--------------" << std::endl;
  // // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // // printf("%d\n", arrLength);
  // h_char_test = (char*) malloc(sizeof(char)*arrLength);
  // cudaMemcpyAsync(h_char_test, res, sizeof(char)*arrLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, arrLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;


  cudaMallocAsync(&instance, arrLength*sizeof(int32_t),0);
  initialize_depth<<<numBlock, BLOCKSIZE>>>(res, instance, arrLength);
  cudaStreamSynchronize(0);
  // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength);
  // cudaMemcpyAsync(h_long_test, instance, sizeof(int32_t)*arrLength, cudaMemcpyDeviceToHost);
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);

  thrust::inclusive_scan(thrust::cuda::par, instance, instance+arrLength, instance);
 

  thrust::transform_if(thrust::cuda::par, instance, instance+arrLength, res, instance, decrease(), is_opening());
  // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength);
  // cudaMemcpyAsync(h_long_test, instance, sizeof(int32_t)*arrLength, cudaMemcpyDeviceToHost);
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);  

  thrust::stable_sort_by_key(thrust::cuda::par, instance, instance + arrLength, res);//
  // h_char_test = (char*) malloc(sizeof(char)*arrLength);
  // cudaMemcpyAsync(h_char_test, res, sizeof(char)*arrLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, arrLength, ROW1);
  // free(h_char_test);

  int numBlockDividedTwo = (numBlock >> 1);
  int arrLengthDiv2 = (arrLength >> 1);
  check_with_next<<<numBlockDividedTwo, BLOCKSIZE>>>(res, instance, arrLengthDiv2); //what is wrong!
  cudaStreamSynchronize(0);

  int sum = thrust::reduce(thrust::cuda::par, instance, instance+arrLengthDiv2);
  bool isCorrect = sum == 0 ? true : false;

  // int nextP2 = arrLength == 1 ? 1 : 1 << (32 - __builtin_clz(arrLength-1));

  // start = clock();
  // uint64_t t1, t2;
  // // cudaEventRecord(gpu_start);
  // bool isCorrect =  matching(res, arrLength, nextP2, numBlock);
   // cudaEventRecord(gpu_stop);
   // cudaEventSynchronize(gpu_stop);
   // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
  // end = clock();

  // allEnd = clock();
  // cudaFreeAsync(res,0);
  // std::cout << "-------------isCorrect--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
  // //printf("%d\n", isCorrect);
  // std::cout << "-------------End isCorrect--------------" << std::endl;

   cudaFreeAsync(res,0);
   cudaFreeAsync(instance,0);

  return isCorrect;
  
}
//INPUT Currectness Check END

__global__
void reduce(int length, int arrLength, char * string, int32_t * arr, int32_t* index_arr, int32_t * res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int32_t i = index; i< length; i+=stride)
  {
    int currentChar = (int)string[i];
    if(i < length && (currentChar == OPENBRACKET || currentChar == OPENBRACE || currentChar == COMMA)){
      if(i != 0) {
        res[arr[i-1]] = arr[length + i - 1]; // Depth
        res[arrLength + arr[i-1]] = arr[i-1]; // Node Count
        res[arrLength*ROW2+arr[i-1]] = index_arr[i];
      }
      else{
        res[arrLength*ROW2] = 0;
        res[0] = 0;
        res[arrLength] = 0;
      } 
    }
  }
}

inline int32_t * sortByDepth(int length, int numBlock, int32_t * arr, int32_t* input_index_d)
{
  //int32_t * res;
  int32_t* tmp;
  //cudaMallocAsync(&res, length*ROW2*sizeof(int32_t),0);
  //cudaMemcpyAsync(res, arr,  length*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW4);
  // cudaMemcpyAsync(h_long_test, (arr), sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost, 0);
  // printf("=================\n");
  // print(h_long_test, length, ROW2);
  // free(h_long_test);
  // printf("______________________________________\n");

  tmp = sort(length, numBlock, arr, input_index_d);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length);
  // cudaMemcpyAsync(h_long_test, input_index_d, sizeof(int32_t)*length, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW1);
  // free(h_long_test);
  // std::cout << "-------------End First Step--------------" << std::endl;
  // R1 Depth
  // R2 Node
  // R3 Depth
  // R4 Node Sorted by Depth
  cudaMemcpyAsync((arr+length*ROW1), tmp, length*ROW1*sizeof(int32_t), cudaMemcpyDeviceToDevice); 
  //cudaFreeAsync(tmp,0);
  return arr;
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
    res[i] = (keys[i] < 0 && i > 0 ) ? arr[arr[i]] : arr[i];
  } 

}

inline int32_t* propagateParentsAndCountChildren(int length, int numBlock, int32_t* arr, int32_t* res)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __builtin_clz(length-1));
  clock_t start, end;
  //int32_t * cudaArr;
  //int32_t * keys_d;
  //int32_t * res;
  //int32_t * index_d;
  thrust::plus<int> op;
  int32_t first_index = -1;

  // printf("=============================\n");
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
  // cudaMemcpyAsync(h_long_test, arr + length*ROW1, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  thrust::inclusive_scan_by_key(thrust::cuda::par, arr + length*ROW3, arr + length*ROW4, arr + length*ROW1, arr + length*ROW1);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, arr + length*ROW1, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);
  
  cudaMemcpyAsync(arr + length*ROW3, arr + length*ROW1,  length*sizeof(int32_t), cudaMemcpyDeviceToDevice);

  thrust::sequence(thrust::cuda::par, arr + length*ROW4, arr + length*ROW5);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW1);
  // cudaMemcpyAsync(h_long_test, arr + length*ROW4, sizeof(int32_t)*length*ROW1, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW1);
  // free(h_long_test);

  //subtract index from  
  thrust::transform_if(thrust::cuda::par, arr + length*ROW1, arr + length*ROW2, arr + length*ROW4, arr + length*ROW3, arr + length*ROW1, op, is_minus());
  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, arr + length*ROW1, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  // *(res+i) = *(cudaArr+*(cudaArr+i)) if key+i < 0
  propagateParents<<<numBlock, BLOCKSIZE>>>(length, arr + length*ROW1, arr + length*ROW3, arr + length*ROW1);    
  cudaStreamSynchronize(0);
  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, arr + length*ROW1, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
  // print(h_long_test, length, ROW2);
  // free(h_long_test);

  //cudaMemcpyAsync(cudaArr, &first_index, sizeof(int32_t), cudaMemcpyHostToDevice);
  //cudaMemcpyAsync(cudaArr+length, &first_index, sizeof(int32_t), cudaMemcpyHostToDevice);

  //cudaMemcpyAsync(keys_d, cudaArr,  length*sizeof(int32_t), cudaMemcpyDeviceToDevice);

  thrust::inclusive_scan_by_key(thrust::cuda::par, arr + length*ROW1, arr + length*ROW2, arr + length*ROW2, arr + length*ROW2);

  // int32_t* h_long_test = (int32_t*)malloc(sizeof(int32_t)*length*ROW2);
  // cudaMemcpyAsync(h_long_test, arr + length*ROW1, sizeof(int32_t)*length*ROW2, cudaMemcpyDeviceToHost);
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
  // printf("==========================================\n");
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

inline int32_t * allocate(int length, int numBlock, int32_t* arr, int32_t* res)
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

inline int32_t scan(int length, int32_t* arr)
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
void generateRes(int length, int resLength, int32_t* arr, int32_t* res, int32_t* index_arr)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  
  for(int32_t i = index; i<length; i+=stride){
    res[arr[length*ROW3 + i]] = arr[length*ROW2 + i];
    res[resLength + arr[length*ROW3 + i]] = index_arr[i];
    // if(i ==0) printf("num children: %d, string index: %d\n", arr[length*ROW2 + i], index_arr[i]);
    if(arr[i] != -1) res[arr[length*ROW3+ arr[i]]+arr[length+ i]] = arr[length*ROW3+ i];
  }
}


int32_t* NewRuntime_Parallel_GPU(char* input_d, int32_t* input_index_d, int length, int & result_size) {
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
  cudaEvent_t gpu_start, gpu_stop;
  float elapsed_time = 0;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  char* h_char_test;
  int32_t* h_long_test;
  allStart = clock();

  start = clock();

  // h_long_test = (int32_t*)malloc(sizeof(int32_t)*attachedLength);
  // cudaMemcpyAsync(h_long_test, input_index_d, sizeof(int32_t)*attachedLength, cudaMemcpyDeviceToHost);
  // print(h_long_test, attachedLength, ROW1);
  // free(h_long_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

  //attacheArr = input;
  //memcpy(attacheArr, input, length*sizeof(char));
  //attacheArr[length] = ',';
  //char* d_attacheArr;
  //cudaMallocAsync(&d_attacheArr, attachedLength*sizeof(char),0);
  //cudaMemcpyAsync(d_attacheArr, input_d, attachedLength*sizeof(char), cudaMemcpyDeviceToDevice);
  
  //cudaMallocAsyncManaged(&attacheArr, attachedLength*sizeof(char),0);
  //cudaMemcpyAsync(attacheArr, input, length*sizeof(char), cudaMemcpyHostToDevice);
  //attacheArr[length] = ',';
  //char* d_sameDepthArr;
  //cudaMallocAsync(&d_sameDepthArr, attachedLength*sizeof(char),0);
  //cudaMemcpyAsync(d_sameDepthArr, input_d, attachedLength*sizeof(char), cudaMemcpyDeviceToDevice);

  // changeDepth<<<numBlock, BLOCKSIZE>>>(attachedLength, d_attacheArr, d_sameDepthArr);
  // cudaStreamSynchronize(0);
  // free(attacheArr);
  // cudaFreeAsync(d_attacheArr,0);
  // end = clock();
  // step1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------First Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_char_test = (char*)malloc(sizeof(char)*attachedLength);
  // cudaMemcpyAsync(h_char_test, input_index_d, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, attachedLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

    //*******************************//
    // size_t l_free = 0;
    // size_t l_Total = 0;
    // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
    // size_t allocated = (l_Total - l_free);
    // std::cout << "Total: " << l_Total << " Free: " << l_free << " Allocated: " << allocated << std::endl;
    //*******************************//
  start = clock();
  // cudaEventRecord(gpu_start);
  int32_t *d_arr;
  int32_t correctDepth;
  correctDepth = findDepthAndCount(attachedLength, numBlock, &arr, input_d);
  // printf("depth: %d\n", correctDepth);
  end = clock();
  // cudaEventRecord(gpu_stop);
  // cudaEventSynchronize(gpu_stop);
  // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);

  step2 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Second Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
  // h_long_test = (int32_t*)malloc(sizeof(int32_t)*attachedLength*ROW3);
  // cudaMemcpyAsync(h_long_test, arr, sizeof(int32_t)*attachedLength*ROW3, cudaMemcpyDeviceToHost);
  // print(h_long_test, attachedLength, ROW2);
  // free(h_long_test);
  // std::cout << "-------------End Second Step--------------" << std::endl;
  int32_t arrLength;
  cudaMemcpyAsync(&arrLength, arr+(attachedLength-2), sizeof(int32_t), cudaMemcpyDeviceToHost);
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
    correct = isCorrect(attachedLength, arr+(attachedLength)*ROW2, input_d);
    if(correct){
      start = clock();
      //printf("         %d\n", attachedLength);

      // cudaMallocAsync(&arr, attachedLength*ROW4*sizeof(int32_t),0);
      // cudaMemcpyAsync(arr, d_arr,  attachedLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      // cudaFreeAsync(d_arr,0);
      // cudaEventRecord(gpu_start);
      // h_long_test = (int32_t *)malloc(sizeof(int32_t)*attachedLength*ROW1);
      // cudaMemcpyAsync(h_long_test, input_index_d, sizeof(int32_t)*attachedLength*ROW1, cudaMemcpyDeviceToHost, 0);
      // print(h_long_test, attachedLength, ROW1);
      // free(h_long_test);
      // printf("______________________________________\n");
      cudaMallocAsync(&res, arrLength*ROW5*sizeof(int32_t),0);
      // thrust::remove_copy_if(thrust::cuda::par, arr, arr+attachedLength, input_d, res, is_not_open_comma());
      // thrust::remove_copy_if(thrust::cuda::par, arr+attachedLength, arr+attachedLength*ROW2, input_d, res+arrLength, is_not_open_comma());
      reduce<<<numBlock, BLOCKSIZE>>>(attachedLength, arrLength, input_d, arr, input_index_d, res); // WHY THAT WIERD NUMBER INDEX 0 OF STRING INDEX ARRAY !!!!!!
      cudaStreamSynchronize(0);
      end = clock();
      cudaMemcpyAsync(input_index_d+1, res+arrLength*ROW2+1, sizeof(int32_t)*arrLength-1, cudaMemcpyDeviceToDevice, 0);
      //printf("open and comma size: %d\n", arrLength);
      // h_long_test = (int32_t*)malloc(sizeof(int32_t)*arrLength*ROW2);
      // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW2);
      // free(h_long_test);
      // std::cout << "-------------End First Step--------------" << std::endl;
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      // h_long_test = (int32_t *)malloc(sizeof(int32_t)*arrLength*ROW1);
      // cudaMemcpyAsync(h_long_test, input_index_d, sizeof(int32_t)*arrLength*ROW1, cudaMemcpyDeviceToHost, 0);
      // print(h_long_test, arrLength, ROW1);
      // free(h_long_test);
      // printf("______________________________________\n");
      step3 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Third Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
      // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW2);
      // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW2);
      // free(h_long_test);
      // std::cout << "-------------End Third Step--------------" << std::endl;
      //cudaFreeAsync(d_sameDepthArr,0);
      int numBlock = (arrLength + BLOCKSIZE - 1) / BLOCKSIZE;

      start = clock();
      // cudaEventRecord(gpu_start);
      cudaMemcpyAsync(arr, res,  arrLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
      fakeRes = sortByDepth(arrLength, numBlock, arr, input_index_d);
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      end = clock();
      step4 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Fourth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
      // h_long_test = (int32_t *)malloc(sizeof(int32_t)*arrLength*ROW2);
      // cudaMemcpyAsync(h_long_test, fakeRes, sizeof(int32_t)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW2);
      // free(h_long_test);
      // std::cout << "-------------End Fourth Step--------------" << std::endl;

      start = clock();
      // cudaEventRecord(gpu_start);
      //R1 Depth
      //R2 node sorted by depth
      //cudaMemcpyAsync(arr, fakeRes,  arrLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice); ///////////////////////
      //cudaFreeAsync(fakeRes,0);
      cudaMemsetAsync(res, -1, arrLength*ROW1*sizeof(int32_t), 0);
      findParents<<<numBlock, BLOCKSIZE>>>( arrLength, arr, res);
      cudaStreamSynchronize(0);
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      end = clock();
      step5 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Fifth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
      // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW1);
      // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW1);
      // free(h_long_test);
      // std::cout << "-------------End Fifth Step--------------" << std::endl;


      start = clock();
      // cudaEventRecord(gpu_start);
      //cudaMemcpyAsync(arr, res,  arrLength*ROW1*sizeof(int32_t), cudaMemcpyDeviceToDevice);////////////////////////
      fakeRes = propagateParentsAndCountChildren(arrLength, numBlock, res, arr);   /////////////// inversed
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      end = clock();
      step6 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Sixth Step--------------" << std::endl;      
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
      //  h_long_test = (int32_t*)malloc(sizeof(int32_t)*arrLength*ROW2);
      //  cudaMemcpyAsync(h_long_test, fakeRes, sizeof(int32_t)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      //  print(h_long_test, arrLength, ROW2);
      //  free(h_long_test);
      // std::cout << "-------------End Sixth Step--------------" << std::endl;
      start = clock();
      // cudaEventRecord(gpu_start);
      //cudaMemcpyAsync(arr, fakeRes,  arrLength*ROW2*sizeof(int32_t), cudaMemcpyDeviceToDevice);///////////////////////
      //cudaFreeAsync(fakeRes,0);
      cudaMemsetAsync(res, -1, arrLength*ROW3*sizeof(int32_t), 0);
      childsNumber<<<numBlock, BLOCKSIZE>>>(arrLength, arr, res);
      cudaStreamSynchronize(0);
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      end = clock();
      step7 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Seventh Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
      // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW3);
      // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*arrLength*ROW3, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW3);
      // free(h_long_test);
      // std::cout << "-------------End Seventh Step--------------" << std::endl;
      
      start = clock();
      // cudaEventRecord(gpu_start);
      //cudaMemcpyAsync(arr, res,  arrLength*ROW3*sizeof(int32_t), cudaMemcpyDeviceToDevice); ////////////////////////////
      fakeRes = allocate(arrLength, numBlock, res, arr); ////////////// inversed
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      end = clock();
      step8 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Eighth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
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
      // cudaEventRecord(gpu_start);
      resLength = scan(arrLength, fakeRes);
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      end = clock();
      scanStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;

      // std::cout << "-------------Scan Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
      // // h_long_test = (int32_t*) malloc(sizeof(int32_t)*arrLength*ROW1);
      // // cudaMemcpyAsync(h_long_test, sumRes, sizeof(int32_t)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // // print(h_long_test, arrLength, ROW1);
      // // free(h_long_test);
      // std::cout << "-------------End Scan Step--------------" << std::endl;
      //cudaMemcpyAsync(&resLength, sumRes + arrLength - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
      start = clock();
      // cudaEventRecord(gpu_start);
      //cudaMemcpyAsync(arr, fakeRes,  arrLength*ROW4*sizeof(int32_t), cudaMemcpyDeviceToDevice); /////////////////////////
      //cudaFreeAsync(sumRes,0);
      //cudaFreeAsync(fakeRes,0);
      cudaFreeAsync(res,0);
      cudaMallocAsync(&res, (arrLength+resLength)*sizeof(int32_t)*ROW2,0);
      cudaMemsetAsync(res, 0, (arrLength+resLength)*sizeof(int32_t)*ROW2, 0);
      generateRes<<<numBlock, BLOCKSIZE>>>(arrLength, arrLength+resLength,  arr, res, input_index_d);
      cudaStreamSynchronize(0);
      // cudaEventRecord(gpu_stop);
      // cudaEventSynchronize(gpu_stop);
      // cudaeventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
      end = clock();
      lastStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Last Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << elapsed_time << std::endl;
      // h_long_test = (int32_t*) malloc(sizeof(int32_t)*(arrLength+resLength)*ROW2);
      // cudaMemcpyAsync(h_long_test, res, sizeof(int32_t)*(arrLength+resLength)*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, (arrLength+resLength), ROW2);
      // free(h_long_test);
      // std::cout << "-------------End Last Step--------------" << std::endl;
      result_size = (arrLength+resLength);
      cudaFreeAsync(arr,0);
      cudaFreeAsync(input_index_d, 0);
      //cudaFreeAsync(res,0);
      allEnd = clock();
      return res;
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
      //exit(0);
      return NULL;  
    }
  }
  else {
    printf("Input invalid\n");
    //exit(0);
    return NULL;
  }
  return NULL;
}
