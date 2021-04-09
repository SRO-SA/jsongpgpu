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
        program=0,
        newstep1=0,
        newstep2=0,
        newstep3=0,
        newstep4=0,
        newstep5=0, 
        newstep6=0, 
        newstep7=0, 
        newstep8=0, 
        newscanStep=0, 
        newlastStep=0,
        newcorrect1=0,
        newcorrect2=0,
        newcorrect3=0,
        newcorrect4=0,
        newprogram=0;

int numRecords = 0;

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

struct pthread_input {
  int textLength;
  int outputSize;
  char* text;
  long* output;
  cudaStream_t *stream;
} pthread_input;


bool isCorrect(int strLength, long* input, char* string, cudaStream_t stream);

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

double runMultipleTimes(double(*func)(char *), char * input){
  double runtime = 0.0;

  for(int i=0; i<RUNTIMES; i++){
    runtime += func(input);
    newprogram+= program;
    program = 0;

  }
  //runtime = runtime/RUNTIMES;
  newstep1= newstep1/RUNTIMES;
  newstep2= newstep2/RUNTIMES;
  newstep3= newstep3/RUNTIMES;
  newstep4= newstep4/RUNTIMES;
  newstep5= newstep5/RUNTIMES; 
  newstep6= newstep6/RUNTIMES; 
  newstep7= newstep7/RUNTIMES; 
  newstep8= newstep8/RUNTIMES;
  newscanStep= newscanStep/RUNTIMES; 
  newlastStep= newlastStep/RUNTIMES;
  newcorrect1= newcorrect1/RUNTIMES;
  newcorrect2= newcorrect2/RUNTIMES;
  newcorrect3= newcorrect3/RUNTIMES;
  newcorrect4= newcorrect4/RUNTIMES;
  newprogram= newprogram/RUNTIMES;
  std::cout << "First step mean time for " << RUNTIMES << " number of runs: " << newstep1 << "ms." << std::endl;
  std::cout << "Second step mean time for " << RUNTIMES << " number of runs: " << newstep2 << "ms." << std::endl;
  std::cout << "Correctenss First step mean time for " << RUNTIMES << " number of runs: " << newcorrect1 << "ms." << std::endl;
  std::cout << "Correctenss Second step mean time for " << RUNTIMES << " number of runs: " << newcorrect2 << "ms." << std::endl;
  std::cout << "Correctenss Third step mean time for " << RUNTIMES << " number of runs: " << newcorrect3 << "ms." << std::endl;
  std::cout << "Correctenss Fourth step mean time for " << RUNTIMES << " number of runs: " << newcorrect4 << "ms." << std::endl;
  std::cout << "Third step mean time for " << RUNTIMES << " number of runs: " << newstep3 << "ms." << std::endl;
  std::cout << "Fourth step mean time for " << RUNTIMES << " number of runs: " <<newstep4 << "ms." << std::endl;
  std::cout << "Fifth step mean time for " << RUNTIMES << " number of runs: " << newstep5 << "ms." << std::endl;
  std::cout << "Sixth step mean time for " << RUNTIMES << " number of runs: " << newstep6 << "ms." << std::endl;
  std::cout << "Seventh step mean time for " << RUNTIMES << " number of runs: " << newstep7 << "ms." << std::endl;
  std::cout << "Eighth step mean time for " << RUNTIMES << " number of runs: " << newstep8 << "ms." << std::endl;
  std::cout << "Scan step mean time for " << RUNTIMES << " number of runs: " << newscanStep << "ms." << std::endl;
  std::cout << "Last step mean time for " << RUNTIMES << " number of runs: " << newlastStep << "ms." << std::endl;

  //std::cout << "Mean time for " << RUNTIMES << " number of runs: " << runtime << "ms." << std::endl;
  std::cout << "Internal Mean time for " << RUNTIMES << " number of runs: " << newprogram << "ms." << std::endl;

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

long *sort(int length, int numBlock, long * arr, cudaStream_t stream)
{
  long* cudaArr;
  cudaMalloc(&cudaArr, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr, length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
  //size_t s;
  //cudaDeviceGetLimit(&s, cudaLimitMallocHeapSize);
  //printf("%d\n", (int)s);
  try
  {
    thrust::device_ptr<long> devArr(cudaArr);
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), cudaArr, cudaArr+length, cudaArr+length);
  }
  catch(thrust::system_error e)
  {
    std::cerr << "Error inside sort: " << e.what() << std::endl;
  }
  catch(thrust::system::detail::bad_alloc e){
    cudaGetLastError();
  }

  
  long *res;
  cudaMalloc(&res, length*ROW1*sizeof(long));
  inv<<<numBlock, BLOCKSIZE, 0, stream>>>(length, cudaArr, res);
  cudaStreamSynchronize(stream);
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

long findDepthAndCount(int length, int numBlock, long* arr, char * string, cudaStream_t stream)
{
  initialize<<<numBlock, BLOCKSIZE, 0, stream>>>(0, length, string, arr);
  cudaStreamSynchronize(stream);
  //gpuErrchk( cudaPeekAtLastError() );
  try
  {
    thrust::inclusive_scan(thrust::cuda::par.on(stream), (arr), (arr) + length, (arr));
    thrust::inclusive_scan(thrust::cuda::par.on(stream), (arr) + length, (arr) + ROW2*length, (arr) + length);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), (arr) + ROW2*length, (arr) + ROW3*length, (arr) + ROW2*length);
  }
  catch(thrust::system_error e)
  {
    std::cerr << "Error inside findDepthAndCount: " << e.what() << std::endl;
  }
  catch(thrust::system::detail::bad_alloc e)
  {
    cudaGetLastError();
  }
  long res;
  cudaMemcpy(&res, (arr)+(ROW2*length)-1, sizeof(long), cudaMemcpyDeviceToHost);
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

long* countNodesRepitition(int length, int numBlock, long* arr, cudaStream_t stream)
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
    cudaStreamSynchronize(stream);
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

bool isCorrect(int strLength, long* input, char* string, cudaStream_t stream)
{
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  long* h_long_test;
  allStart = clock();
  long arrLength;
  cudaMemcpy(&arrLength, input + strLength - 1, sizeof(long), cudaMemcpyDeviceToHost);
  arrLength++;
  int numBlock = ((arrLength) + BLOCKSIZE - 1) / BLOCKSIZE;
  char* res;
  // std::cout << "-------------fdsfdfdsfdsfdsfsdf First Step--------------" << pthread_self()<< std::endl;
  // h_char_test = (char*) malloc(sizeof(char)*strLength);
  // cudaMemcpy(h_char_test, string, sizeof(char)*strLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, strLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;
  start = clock();
  cudaMalloc(&res, arrLength*sizeof(char));
  extract<<<numBlock, BLOCKSIZE, 0, stream>>>(strLength, arrLength, string, input, res);
  cudaStreamSynchronize(stream);
  end = clock();
  correct1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  // std::cout << "-------------Curretness First Step--------------" << std::endl;
  // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  // h_char_test = (char*) malloc(sizeof(char)*strLength);
  // cudaMemcpy(h_char_test, string, sizeof(char)*strLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, strLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;

  long* arr;
  start = clock();
  cudaMalloc(&arr, arrLength*sizeof(long));
  initialize<<<numBlock, BLOCKSIZE, 0, stream>>>(1, arrLength, res, arr);
  cudaStreamSynchronize(stream);
  //gpuErrchk( cudaPeekAtLastError() );

  long* longRes;
  try
  {
    thrust::inclusive_scan(thrust::cuda::par.on(stream), arr, arr + arrLength, arr);
  }
  catch(thrust::system_error e)
  {
    std::cerr << "Error inside isCorrect: " << e.what() << std::endl;
  }
  catch(thrust::system::detail::bad_alloc e){
    cudaGetLastError();
  }
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
  longRes = countNodesRepitition(arrLength, numBlock, arr, stream);
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
  checkCurrectenss<<<numBlock, BLOCKSIZE, 0, stream>>>(arrLength, res, (longRes+arrLength), arr);
  cudaStreamSynchronize(stream);
  try
  {
    thrust::inclusive_scan(thrust::cuda::par.on(stream), arr, arr + arrLength, arr);
  }
  catch(thrust::system_error e)
  {
    std::cerr << "Error inside checkCurrectenss: " << e.what() << std::endl;
  }
  catch(thrust::system::detail::bad_alloc e){
    cudaGetLastError();
  }

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

long * sortByDepth(int length, int numBlock, long * arr, cudaStream_t stream)
{
  long * res;
  long* tmp;
  cudaMalloc(&res, length*ROW2*sizeof(long));
  cudaMemcpy(res, arr,  length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
  tmp = sort(length, numBlock, arr, stream);
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

long* propagateParentsAndCountChildren(int length, int numBlock, long* arr, cudaStream_t stream)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __builtin_clz(length-1));
  long * cudaArr;
  long * res;
  cudaMalloc(&cudaArr, length*ROW2*sizeof(long));
  cudaMalloc(&res, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*sizeof(long)*ROW1, cudaMemcpyDeviceToDevice);
  //cudaMemcpy(cudaArr, arr,  length*sizeof(long)*ROW1, cudaMemcpyDeviceToDevice);
  //std::cout << "-------------Seventh Step--------------" << std::endl;
  long * h_long_test = (long*) malloc(sizeof(long)*length*ROW1);
  cudaMemcpy(h_long_test, arr, sizeof(long)*length*ROW1, cudaMemcpyDeviceToHost);
  //print(h_long_test, length, ROW1);
  free(h_long_test);
  //std::cout << "-------------End Seventh Step--------------" << std::endl;
  int i = -1;
  for(int n = nextP2*2; n>1; n=n>>1){
    propagateParentsAndCountChildrenStep<<<numBlock, BLOCKSIZE, 0, stream>>>(length, cudaArr, i, res);
    cudaStreamSynchronize(stream);
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

long * allocate(int length, int numBlock, long* arr, cudaStream_t stream)
{
  long * cudaArr;
  cudaMalloc(&cudaArr, length*ROW4*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*ROW3*sizeof(long), cudaMemcpyDeviceToDevice);
  cudaMemcpy(cudaArr+length*ROW3+1, arr+length*ROW2,  (length*ROW1-1)*sizeof(long), cudaMemcpyDeviceToDevice);
  addOne<<<numBlock, BLOCKSIZE, 0, stream>>>(length, cudaArr);
  cudaStreamSynchronize(stream);
  try
  {
    thrust::inclusive_scan(thrust::cuda::par.on(stream), cudaArr+ROW3*length, cudaArr + ROW4*length, cudaArr+ROW3*length);
  }
  catch(thrust::system_error e)
  {
    std::cerr << "Error inside allocate: " << e.what() << std::endl;
  }
  catch(thrust::system::detail::bad_alloc e){
    cudaGetLastError();
  }

  return cudaArr;
}

long * scan(int length, long* arr, cudaStream_t stream)
{
  long * cudaArr;
  //long * res;
  cudaMalloc(&cudaArr, length*ROW4*sizeof(long));
  //cudaMalloc(&res, length*ROW1*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*ROW4*sizeof(long), cudaMemcpyDeviceToDevice); 
  try
  {
    thrust::inclusive_scan(thrust::cuda::par.on(stream), cudaArr+ROW2*length, cudaArr + ROW3*length, cudaArr);
  }
  catch(thrust::system_error e)
  {
    std::cerr << "Error inside scan: " << e.what() << std::endl;
  }
  catch(thrust::system::detail::bad_alloc e){
    cudaGetLastError();
  }

  //cudaFree(cudaArr);
  return cudaArr;  
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


void *NewRuntime_Parallel_GPU(void* inp) {
  cudaSetDevice(0);
  size_t s = (1 << 30);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, s);
  struct pthread_input* input = (struct pthread_input*) inp;
  cudaStream_t stream = *(input->stream);
  //cudaProfilerStart();
  int attachedLength = input->textLength;
  int numBlock = (attachedLength + BLOCKSIZE - 1) / BLOCKSIZE;
  long* res;
  long* fakeRes;
  long* arr;
  char* attacheArr;
  clock_t start, end, allStart, allEnd;
  char* h_char_test;
  long* h_long_test;
  printf("0:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        //*******************************//
        size_t l_free = 0;
        size_t l_Total = 0;
        cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
        size_t allocated = (l_Total - l_free);
        std::cout << "Error: " << error_id << "Total: " << l_Total << " Free: " << l_free << " Allocated: " << allocated << std::endl;
        //*******************************//
  allStart = clock();
  start = clock();
  pthread_t self;
  self = pthread_self();
  //input->text[attachedLength-1] = ',';
  printf("1:\t%s\n", cudaGetErrorString(cudaGetLastError()));
  attacheArr = (char*) malloc(sizeof(char)*attachedLength);
  printf("2\t%s\n", cudaGetErrorString(cudaGetLastError()));
  memcpy(attacheArr, input->text, attachedLength*sizeof(char));
  printf("3:\t%s\n", cudaGetErrorString(cudaGetLastError()));
  attacheArr[attachedLength-1] = ',';
  char* d_attacheArr;
  char* d_sameDepthArr;
  printf("4:\t%s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMalloc(&d_attacheArr, attachedLength*sizeof(char));
  printf("5:\t%s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMalloc(&d_sameDepthArr, attachedLength*sizeof(char));
  //cudaStreamSynchronize(stream);
  //cudaMallocManaged(&attacheArr, attachedLength*sizeof(char));
  //cudaMemcpy(attacheArr, input, length*sizeof(char), cudaMemcpyHostToDevice);
  //attacheArr[length] = ',';
  printf("6:\t%s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(d_attacheArr, attacheArr, attachedLength*sizeof(char), cudaMemcpyHostToDevice);
  printf("7:\t%s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(d_sameDepthArr, d_attacheArr, attachedLength*sizeof(char), cudaMemcpyHostToDevice);
  printf("8\t%s\n", cudaGetErrorString(cudaGetLastError()));
  // std::cout << "-------------First Step--------------" << std::endl;
  // printString(attacheArr, attachedLength, ROW1);
  // h_char_test = (ch  // std::cout << "-------------First Step--------------" << std::endl;
  // printString(attacheArr, attachedLength, ROW1);
  // h_char_test = (char*)malloc(sizeof(char)*attachedLength);
  // std::cout << "-------------____________________--------------" << std::endl;
  // cudaMemcpy(h_char_test, d_sameDepthArr, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, attachedLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;ar*)malloc(sizeof(char)*attachedLength);
  // std::cout << "-------------____________________--------------" << std::endl;
  // cudaMemcpy(h_char_test, d_sameDepthArr, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
  // printString(h_char_test, attachedLength, ROW1);
  // free(h_char_test);
  // std::cout << "-------------End First Step--------------" << std::endl;
  changeDepth<<<numBlock, BLOCKSIZE, 0, stream>>>(attachedLength, d_attacheArr, d_sameDepthArr);
  cudaStreamSynchronize(stream);
  free(attacheArr);
  cudaFree(d_attacheArr);
  end = clock();
  step1 += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
  //  std::cout << "-------------First Step--------------" << std::endl;
  //  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //  h_char_test = (char*)malloc(sizeof(char)*attachedLength);
  //  cudaMemcpy(h_char_test, d_sameDepthArr, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
  //  printString(h_char_test, attachedLength, ROW1);
  //  free(h_char_test);
  //  std::cout << "-------------End First Step--------------" << std::endl;
  start = clock();
  long *d_arr;
  long correctDepth;
  cudaMalloc(&d_arr, attachedLength*ROW3*sizeof(long));
  correctDepth = findDepthAndCount(attachedLength, numBlock, d_arr, d_sameDepthArr, stream);
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
    // std::cout << "-------------Curretness First Step--------------" << std::endl;
    // h_char_test = (char*) malloc(sizeof(char)*attachedLength);
    // cudaMemcpy(h_char_test, d_sameDepthArr, sizeof(char)*attachedLength, cudaMemcpyDeviceToHost);
    // printString(h_char_test, attachedLength, ROW1);
    // free(h_char_test);
    // std::cout << "-------------End First Step--------------" << std::endl;
    correct = isCorrect(attachedLength, d_arr+(attachedLength)*ROW2, d_sameDepthArr, stream);
    if(correct){
      start = clock();
      // std::cout << "-------------Third Step--------------" << std::endl;
      // h_long_test = (long*) malloc(sizeof(long)*attachedLength*ROW2);
      // cudaMemcpy(h_long_test, d_arr, sizeof(long)*attachedLength*ROW2, cudaMemcpyDeviceToHost);
      // print(h_long_test, attachedLength, ROW2);
      // free(h_long_test);
      // std::cout << "-------------End Third Step--------------" << std::endl;
      cudaMalloc(&arr, attachedLength*ROW4*sizeof(long));
      cudaMalloc(&res, arrLength*ROW4*sizeof(long));
      cudaMemcpy(arr, d_arr,  attachedLength*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
      reduce<<<numBlock, BLOCKSIZE, 0, stream>>>(attachedLength, arrLength, d_sameDepthArr, arr, res);
      cudaStreamSynchronize(stream);
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
      cudaMemcpy(arr, res,  arrLength*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);
      fakeRes = sortByDepth(arrLength, numBlock, arr, stream);
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
      findParents<<<numBlock, BLOCKSIZE, 0, stream>>>( arrLength, arr, res);
      cudaStreamSynchronize(stream);
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
      fakeRes = propagateParentsAndCountChildren(arrLength, numBlock, arr, stream);
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

      cudaDeviceSynchronize();

      childsNumber<<<numBlock, BLOCKSIZE, 0, stream>>>(arrLength, arr, res);
      cudaStreamSynchronize(stream);
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
      fakeRes = allocate(arrLength, numBlock, arr, stream);
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
      sumRes = scan(arrLength, fakeRes, stream);
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
      cudaMemcpy(arr, fakeRes,  arrLength*ROW4*sizeof(long), cudaMemcpyDeviceToDevice);
      cudaFree(sumRes);
      cudaFree(fakeRes);
      cudaFree(res);
      cudaMalloc(&res, (arrLength+resLength)*sizeof(long));
      cudaMemset(res, 0, (arrLength+resLength)*sizeof(long));
      //cudaStreamSynchronize(stream);
      cudaDeviceSynchronize();
      size_t free, total;
      //cudaMemGetInfo 	(&free,&total);
      //printf("free: %d, total: %d", (int)free, (int)total);
      generateRes<<<numBlock, BLOCKSIZE, 0, stream>>>(arrLength,  arr, res);
      printf("9\t%s\n", cudaGetErrorString(cudaGetLastError())); //Synchronize give error!!!!!!!!!!!!!!!!!!!

      cudaStreamSynchronize(stream);
      end = clock();
      lastStep += ((double)(end-start)/CLOCKS_PER_SEC)*1000;
      // std::cout << "-------------Last Step--------------" << std::endl;
      // std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      // h_long_test = (long*) malloc(sizeof(long)*(arrLength+resLength)*ROW1);
      // cudaMemcpy(h_long_test, res, sizeof(long)*(arrLength+resLength)*ROW1, cudaMemcpyDeviceToHost);
      // print(h_long_test, (arrLength+resLength), ROW1);
      // free(h_long_test);
      // std::cout << "-------------End Last Step--------------" << std::endl;
      input->outputSize = (arrLength+resLength);
      input->output = (long *) malloc(sizeof(long)*(input->outputSize));
      cudaMemcpy(input->output, res, sizeof(long)*(input->outputSize), cudaMemcpyDeviceToHost);

      cudaFree(arr);
      cudaFree(res);
      allEnd = clock();
      //cudaProfilerStop();    

      //printf("program: %f\n", program);
      return NULL;
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
  return NULL;
}

char **loadMultipleRecords(char* name, int * number, int** recordsLength){
  FILE * newfile;
  char * line= NULL;
  size_t len = 0;
  ssize_t read;
  newfile = fopen(name, "r");

  if(newfile == NULL) exit(EXIT_FAILURE);
  int lineNum = 0;
  while((read = getline(&line, &len, newfile))  != -1) lineNum++;
  fclose(newfile);
  char** texts = (char **)malloc(lineNum*sizeof(char *));
  *recordsLength = (int *)malloc(sizeof(int)*lineNum);
  int i = 0;
  line= NULL;
  len = 0;
  newfile = fopen(name, "r");
  if(newfile == NULL) exit(EXIT_FAILURE);
  while((read = getline(&line, &len, newfile))  != -1){
    *(texts+i) = (char*) malloc(sizeof(char)*read);
    memcpy(*(texts+i), (char*)line, sizeof(char)*read);
    *((*recordsLength)+i) = (int)read;
    //printString((char*)texts[i], *((*recordsLength)+i), ROW1);

    i++;
  }
  *number = lineNum;
  fclose(newfile);
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

double batchMode(char* filename)
{
  clock_t allStart, allEnd;
  char ** texts;
  int numTexts;
  int* recordsLength = NULL;
  texts = loadMultipleRecords(filename, &numTexts, &recordsLength);
  const int numThreads = numTexts > 1 ? 1 : numTexts;
  double GPUparallelTime = 0;
  pthread_t threads[numThreads];
  struct pthread_input threadsInput[numTexts];
  cudaStream_t streams[numThreads];
  numRecords = numTexts;
  char * text;
  //for(unsigned int i=0; i<numTexts; i++) {printString((char*)texts[i], recordsLength[i], ROW1);}
  unsigned int n=0;
  allStart = clock();
  for(unsigned int i=0; i<numThreads; i++) {cudaStreamCreate(&streams[i]);}
  while(n < numTexts){
    printf("1:____________%d____________\n", n);
    for(unsigned int k=0; k<numThreads; k++){
      int index = n+k;
      if(index < numTexts){
        text = texts[index];
        //recordsLength[index] = strlen(texts[index]);
        threadsInput[index].stream = &streams[k];
        //printString(text, recordsLength[index], ROW1);
        //cudaStreamCreate(threadsInput[index].stream);
        threadsInput[index].text = (char*)malloc(sizeof(char)*recordsLength[index]);
        memcpy(threadsInput[index].text, text, sizeof(char)*recordsLength[index]);
        threadsInput[index].textLength = recordsLength[index];
        free(text);
        printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        //printString(threadsInput[index].text, recordsLength[index], ROW1);
        if(pthread_create(&threads[k], NULL, NewRuntime_Parallel_GPU, (void*)&threadsInput[index])){
          printf("Error creating threads\n");
          return 0;
        }
      }
    }
    for (unsigned int i = 0; i < numThreads; i++) {
      int index = n+i;
      if(index<numTexts){
        if(pthread_join(threads[i], NULL)) {
            printf("Error joining threads\n");
            return 2;
        }
        free(threadsInput[index].text);
        //cudaStreamDestroy(*(threadsInput[index].stream));
      }
      printf("2:____________%d____________\n", n);
    }

    step1= step1/numThreads;
    step2= step2/numThreads;
    step3= step3/numThreads;
    step4= step4/numThreads;
    step5= step5/numThreads;
    step6= step6/numThreads;
    step7= step7/numThreads;
    step8= step8/numThreads;
    scanStep= scanStep/numThreads; 
    lastStep= lastStep/numThreads;
    correct1= correct1/numThreads;
    correct2= correct2/numThreads;
    correct3= correct3/numThreads;
    correct4= correct4/numThreads;

    newstep1+= step1;
    newstep2+= step2;
    newstep3+= step3;
    newstep4+= step4;
    newstep5+= step5;
    newstep6+= step6;
    newstep7+= step7;
    newstep8+= step8;
    newscanStep+= scanStep; 
    newlastStep+= lastStep;
    newcorrect1+= correct1;
    newcorrect2+= correct2;
    newcorrect3+= correct3;
    newcorrect4+= correct4;

    step1= 0;
    step2= 0;
    step3= 0;
    step4= 0;
    step5= 0;
    step6= 0;
    step7= 0;
    step8= 0;
    scanStep= 0; 
    lastStep= 0;
    correct1= 0;
    correct2= 0;
    correct3= 0;
    correct4= 0;
    
    n = n+numThreads;
  }
  allEnd = clock();
  program += ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000;
  for(unsigned int i=0; i<numThreads; i++) {cudaStreamDestroy(*(threadsInput[i].stream));}
  //print(threadsInput[0].output, threadsInput[0].outputSize, ROW1);
  //for(int i=0; i<numTexts; i++){
    //print(threadsInput[i].output, threadsInput[i].outputSize, ROW1);
  //}


  return (GPUparallelTime/CLOCKS_PER_SEC)*1000;
  
}

double singleMode(char* filename){
  int length;
  char* input = loadFile(&length);
  if(input){
    //std::cout <<  length << std::endl;
    double GPUparallelTime = 0;
    //NewRuntime_Parallel_GPU(input, length);
    //GPUparallelTime = NewRuntime_Parallel_GPU(input, length);
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
  if (argv[1] != NULL){
    if( strcmp(argv[1], "-b") == 0 && argv[2] != NULL){
      std::cout << "Batch mode..." << std::endl;
      result = runMultipleTimes(batchMode, argv[2]);
    }
    if (strcmp(argv[1], "-s") == 0 && argv[2] != NULL){
      std::cout << "Single mode..." << std::endl;
      result = runMultipleTimes(batchMode, argv[2]);
    }
    else std::cout << "Command should be like '-b/-s [file path]/[file name]'" << std::endl;
  }
  else{
    std::cout << "Please select (batch: -b, single -s): " << std::endl;
  }
  return 0;
}