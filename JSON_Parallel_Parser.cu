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
  long* cudaArr;
  cudaMalloc(&cudaArr, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr, length*ROW2*sizeof(long), cudaMemcpyHostToDevice);
  thrust::device_ptr<long> devArr(cudaArr);
  thrust::stable_sort_by_key(thrust::cuda::par, cudaArr, cudaArr+length, cudaArr+length);
  long *res;
  cudaMalloc(&res, length*ROW1*sizeof(long));
  inv<<<numBlock, BLOCKSIZE>>>(length, cudaArr, res);
  cudaDeviceSynchronize();
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
  // h_char_test = (char*) malloc(sizeof(char)*arrLength);
  // cudaMemcpy(h_char_test, res, sizeof(char)*arrLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, arrLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

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
  // std::cout << "-------------Curretness Second Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (long*) malloc(sizeof(long)*arrLength);
  // cudaMemcpy(h_long_test, arr, sizeof(long)*arrLength, cudaMemcpyDeviceToHost);
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  // std::cout << "-------------End Second Step--------------" << std::endl;

  start = clock();
  longRes = countNodesRepitition(arrLength, numBlock, arr);
  end = clock();
  correct3 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Curretness Third Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (long *)malloc(sizeof(long)*arrLength);
  // cudaMemcpy(h_long_test, longRes, sizeof(long)*arrLength, cudaMemcpyDeviceToHost); 
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  // std::cout << "-------------End Third Step--------------" << std::endl;

  start = clock();
  checkCurrectenss<<<numBlock, BLOCKSIZE>>>(arrLength, res, (longRes+arrLength), arr);
  cudaDeviceSynchronize();
  thrust::inclusive_scan(thrust::cuda::par, arr, arr + arrLength, arr);
  end = clock();
  correct4 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Curretness Fourth Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (long *)malloc(sizeof(long)*arrLength);
  // cudaMemcpy(h_long_test, arr, sizeof(long)*arrLength, cudaMemcpyDeviceToHost); 
  // print(h_long_test, arrLength, ROW1);
  // free(h_long_test);
  // std::cout << "-------------End Fourth Step--------------" << std::endl;

  allEnd = clock();

  cudaFree(res);
  cudaFree(longRes);
  long isCorrect;
  cudaMemcpy(&isCorrect, arr+arrLength-1, sizeof(long), cudaMemcpyDeviceToHost);
  cudaFree(arr);
  // std::cout << "-------------isCorrect--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000 << std::endl;
  // printf("%d\n", isCorrect == arrLength);
  // std::cout << "-------------End isCorrect--------------" << std::endl;
  return isCorrect == arrLength;
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
void propagateParentsAndCountChildrenStep(int length, long* arr, int i, long* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long j=index; j<length; j+=stride){
    if( i== -1){
      res[j] = arr[j];
      if(j != 0) res[length + j] = 1;
      else res[length + j] = -1;
    }
    if(i > -1){
      int pow2 = 1<<i;
      if(j >= pow2){
        if(arr[j] == -1 && arr[j - pow2] != -1) {
          res[j] = arr[j - pow2];
        }
        else{
          res[j] = arr[j];
        }
        if(!((arr[j] != -1 && arr[j - pow2] == -1) || (arr[j] != -1 && arr[j - pow2] != -1 && arr[j] != arr[j - pow2]))){
          res[length + j] = arr[length + j - pow2] + arr[length + j];
        }
        else {
          res[length + j] = arr[length + j];
        }
      }
      else{
        res[j] = arr[j];
        res[length + j] = arr[length + j];
      }
    }
  }  
}

long* propagateParentsAndCountChildren(int length, int numBlock, long* arr)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __builtin_clz(length-1));
  long * cudaArr;
  long * res;
  cudaMalloc(&cudaArr, length*ROW2*sizeof(long));
  cudaMalloc(&res, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*sizeof(long), cudaMemcpyDeviceToDevice);
  int i = -1;
  for(int n = nextP2*2; n>1; n=n>>1){
    propagateParentsAndCountChildrenStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, res);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaArr, res,  length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);    
    i+=1;
  }
  cudaFree(cudaArr);
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


double NewRuntime_Parallel_GPU(char* input, int length) {
  cudaProfilerStart();
  int attachedLength = length + 1;
  int numBlock = (attachedLength + BLOCKSIZE - 1) / BLOCKSIZE;
  long* res;
  long* fakeRes;
  long* arr;
  char* attacheArr;
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  long* h_long_test;
  allStart = clock();

  start = clock();

  attacheArr = (char*) malloc(sizeof(char)*attachedLength);
  memcpy(attacheArr, input, length*sizeof(char));
  attacheArr[length] = ',';
  char* d_attacheArr;
  cudaMalloc(&d_attacheArr, attachedLength*sizeof(char));
  cudaMemcpy(d_attacheArr, attacheArr, attachedLength*sizeof(char), cudaMemcpyHostToDevice);
  
  //cudaMallocManaged(&attacheArr, attachedLength*sizeof(char));
  //cudaMemcpy(attacheArr, input, length*sizeof(char), cudaMemcpyHostToDevice);
  //attacheArr[length] = ',';
  char* d_sameDepthArr;
  cudaMalloc(&d_sameDepthArr, attachedLength*sizeof(char));
  cudaMemcpy(d_sameDepthArr, attacheArr, attachedLength*sizeof(char), cudaMemcpyHostToDevice);

  changeDepth<<<numBlock, BLOCKSIZE>>>(attachedLength, d_attacheArr, d_sameDepthArr);
  cudaDeviceSynchronize();
  free(attacheArr);
  cudaFree(d_attacheArr);
  end = clock();
  step1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------First Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_char_test = (char*)malloc(sizeof(char)*attachedLength);
  // cudaMemcpy(h_char_test, d_sameDepthArr, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, attachedLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

  start = clock();
  long *d_arr;
  long correctDepth;
  correctDepth = findDepthAndCount(attachedLength, numBlock, &d_arr, d_sameDepthArr);
  end = clock();
  step2 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Second Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_long_test = (long*)malloc(sizeof(long)*attachedLength*ROW3);
  // cudaMemcpy(h_long_test, d_arr, sizeof(long)*attachedLength*ROW3, cudaMemcpyDeviceToHost);
  // print(h_long_test, attachedLength, ROW3);
  // free(h_long_test);
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
      // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW2);
      // cudaMemcpy(h_long_test, res, sizeof(long)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW2);
      // free(h_long_test);
      // std::cout << "-------------End Third Step--------------" << std::endl;
      cudaFree(d_sameDepthArr);
      cudaFree(d_arr);
      int numBlock = (arrLength + BLOCKSIZE - 1) / BLOCKSIZE;

      start = clock();
      cudaMemcpy(arr, res,  arrLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      fakeRes = sortByDepth(arrLength, numBlock, arr);
      end = clock();
      step4 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Fourth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // h_long_test = (long *)malloc(sizeof(long)*arrLength*ROW2);
      // cudaMemcpy(h_long_test, fakeRes, sizeof(long)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW2);
      // free(h_long_test);
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
      // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW1);
      // cudaMemcpy(h_long_test, res, sizeof(long)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW1);
      // free(h_long_test);
      // std::cout << "-------------End Fifth Step--------------" << std::endl;
      
      start = clock();
      cudaMemcpy(arr, res,  arrLength*ROW1*sizeof(long), cudaMemcpyDeviceToDevice);
      fakeRes = propagateParentsAndCountChildren(arrLength, numBlock, arr);
      end = clock();
      step6 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Sixth Step--------------" << std::endl;      
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // h_long_test = (long*)malloc(sizeof(long)*arrLength*ROW2);
      // cudaMemcpy(h_long_test, fakeRes, sizeof(long)*arrLength*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW2);
      // free(h_long_test);
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
      // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW3);
      // cudaMemcpy(h_long_test, res, sizeof(long)*arrLength*ROW3, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW3);
      // free(h_long_test);
      // std::cout << "-------------End Seventh Step--------------" << std::endl;
      
      start = clock();
      cudaMemcpy(arr, res,  arrLength*ROW3*sizeof(long), cudaMemcpyDeviceToDevice);
      fakeRes = allocate(arrLength, numBlock, arr);
      end = clock();
      step8 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Eighth Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW4);
      // cudaMemcpy(h_long_test, fakeRes, sizeof(long)*arrLength*ROW4, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW4);
      // free(h_long_test);
      // std::cout << "-------------End Eighth Step--------------" << std::endl;

      long* sumRes;
      cudaMalloc(&sumRes, arrLength*ROW1*sizeof(long));
      start = clock();
      sumRes = scan(arrLength, fakeRes);
      end = clock();
      scanStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Scan Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // h_long_test = (long*) malloc(sizeof(long)*arrLength*ROW1);
      // cudaMemcpy(h_long_test, sumRes, sizeof(long)*arrLength*ROW1, cudaMemcpyDeviceToHost);
      // print(h_long_test, arrLength, ROW1);
      // free(h_long_test);
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
      // h_long_test = (long*) malloc(sizeof(long)*(arrLength+resLength)*ROW1);
      // cudaMemcpy(h_long_test, res, sizeof(long)*(arrLength+resLength)*ROW1, cudaMemcpyDeviceToHost);
      // print(h_long_test, (arrLength+resLength), ROW1);
      // free(h_long_test);
      // std::cout << "-------------End Last Step--------------" << std::endl;
      cudaFree(arr);
      cudaFree(res);
      allEnd = clock();
      cudaProfilerStop();    
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

char **loadMultipleFiles(int length, char** names, int * filesLength){
  char** texts = (char **)malloc(length*sizeof(char *));
  for(int i = 0; i< length; i++){
    FILE * f = fopen(names[i], "r");
    if(f){
      fseek(f, 0, SEEK_END);
      filesLength[i] = ftell(f);
      fseek(f, 0, SEEK_SET);
      texts[i] = (char *)malloc(sizeof(char) * filesLength[i]);
      if(texts[i]) {
          fread(texts[i], 1, filesLength[i], f);
      }
      fclose(f);
    }
  }
  return texts;
}

char* loadFile(int* fileLength){
  char * input = 0;
  long length;
  // Long input
  //FILE * f = fopen("./inputs/Long.txt", "r");

  // Long 2 times
  //FILE * f = fopen("./inputs/Long_2.txt", "r");

  // Long 4 times
  //FILE * f = fopen("./inputs/Long_4.txt", "r");

  // Long 8 times
  //FILE * f = fopen("./inputs/Long_8.txt", "r");

  // Long 16 times
  //FILE * f = fopen("./inputs/Long_16.txt", "r");

  // Long 32 times
  //FILE * f = fopen("./inputs/Long_32.txt", "r");

  // Long 64 times
  //FILE * f = fopen("./inputs/Long_64.txt", "r");

  // Long 128 times
  FILE * f = fopen("./inputs/Long_128.txt", "r");

  // Long 256 times
  //FILE * f = fopen("./inputs/Long_256.txt", "r");

  // Long 512 times
  //FILE * f = fopen("./inputs/Long_512.txt", "r");

  // Long 1024 times
  //FILE * f = fopen("./inputs/Long_1024.txt", "r");

  // Base input
  //FILE * f = fopen("./inputs/Base.txt", "r");

  // Author input
  //FILE * f = fopen("./inputs/Author.txt", "r");
  
  // False input
  //FILE * f = fopen("./inputs/False.txt", "r");
  
  // One input
  //FILE * f = fopen("./inputs/One.txt", "r");

  if(f){
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    input = (char *)malloc(sizeof(char) * length);
    if(input) {
        fread(input, 1, length, f);
    }
    fclose(f);
    *fileLength = length;
    return input;
  }
  return 0;
}

double batchMode()
{
  char ** texts;
  char ** names = (char**) malloc(FILESCOUNT*sizeof(char *));
  int * fileLength = (int *)malloc(FILESCOUNT*sizeof(int));
  for(int i = 0; i< FILESCOUNT; i++){
    names[i] = (char *)malloc(NAMELENGTH*sizeof(char));
    strcpy(names[i], FILENAMES[i]);
  }
  texts = loadMultipleFiles(FILESCOUNT, names, fileLength);
  double GPUparallelTime = 0;
  //NewRuntime_Parallel_GPU(texts[0], fileLength[0]);
  if(texts){
    for(int i=0; i<FILESCOUNT; i++){
      //std::cout <<  fileLength[i] << std::endl;
      GPUparallelTime += NewRuntime_Parallel_GPU(texts[i], fileLength[i]);
      //std::cout << "Parallel GPU time elapsed: " << std::setprecision (17) << (GPUparallelTime/CLOCKS_PER_SEC)*1000 << "ms." << std::endl;
    }
    return (GPUparallelTime/CLOCKS_PER_SEC)*1000;
  }
  else{
    printf("Cannot read file\n");
    return -1;
  }
}

double singleMode(){
  int length;
  char* input = loadFile(&length);
  if(input){
    //std::cout <<  length << std::endl;
    double GPUparallelTime = 0;
    //NewRuntime_Parallel_GPU(input, length);
    GPUparallelTime = NewRuntime_Parallel_GPU(input, length);
    //std::cout << "Parallel GPU time elapsed: " << std::setprecision (17) << (GPUparallelTime/CLOCKS_PER_SEC)*1000 << "ms." << std::endl;
    return (GPUparallelTime/CLOCKS_PER_SEC)*1000;
  }
  else{
    printf("Cannot read file\n");
    return -1;
  }
}

int main(int argc, char **argv)
{
  double result;
  if (argv[1] != NULL && strcmp(argv[1], "-b") == 0){
    std::cout << "Batch mode..." << std::endl;
    result = runMultipleTimes(batchMode);
  }
  else{
    std::cout << "Single mode..." << std::endl;
    result = runMultipleTimes(singleMode);
  }
  return 0;
}