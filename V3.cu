#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <iomanip>
#include <intrin.h>
#include "cuda_profiler_api.h"
#include "DyckNew_Parallel_GPU.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <string.h>

const char const * FILENAMES[]={"./inputs/Long.txt", "./inputs/Long_2.txt", "./inputs/Long_4.txt", "./inputs/Long_8.txt"};

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
  cudaMallocManaged(&res, length*ROW1*sizeof(long));
  inv<<<numBlock, BLOCKSIZE>>>(length, cudaArr, res);
  cudaDeviceSynchronize();
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

long *findDepthAndCount(int length, int numBlock, char * string)
{
  long * arr;
  cudaMallocManaged(&arr, length*ROW3*sizeof(long));
  initialize<<<numBlock, BLOCKSIZE>>>(0, length, string, arr);
  cudaDeviceSynchronize();
  thrust::inclusive_scan(thrust::cuda::par, arr, arr + length, arr);
  thrust::inclusive_scan(thrust::cuda::par, arr + length, arr + ROW2*length, arr + length);
  thrust::exclusive_scan(thrust::cuda::par, arr + ROW2*length, arr + ROW3*length, arr + ROW2*length);

  if(arr[ROW2*length - 1] == 0){
    return arr;
  }
  arr[0] = -1;
  return arr;
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
  int nextP2 = length == 1 ? 1 : 1 << (32 - __lzcnt(length-1));
  long * cudaArr;
  long * res;
  cudaMallocManaged(&cudaArr, length*ROW2*sizeof(long));
  cudaMallocManaged(&res, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*sizeof(long), cudaMemcpyHostToDevice);
  int i = -1;

  for(int n = nextP2*2; n>1; n=n>>1){
    countNodesRepititionStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, res);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaArr, res,  length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);    
    i+=1;
  }
  return (res + length);
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
  allStart = clock();
  int arrLength = input[strLength-1] + 1;
  int numBlock = ((arrLength) + BLOCKSIZE - 1) / BLOCKSIZE;
  char* res;
  cudaMallocManaged(&res, arrLength*sizeof(char));
  start = clock();
  extract<<<numBlock, BLOCKSIZE>>>(strLength, arrLength, string, input, res);
  cudaDeviceSynchronize();
  end = clock();
  //std::cout << "-------------Curretness First Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //printString(res, arrLength, ROW1);
  //std::cout << "-------------End First Step--------------" << std::endl;

  long* arr;
  char* braceBrackets;
  cudaMallocManaged(&braceBrackets, arrLength*sizeof(char));
  cudaMemcpy(braceBrackets, res, arrLength*sizeof(char), cudaMemcpyHostToDevice);
  cudaMallocManaged(&arr, arrLength*sizeof(long));
  start = clock();
  initialize<<<numBlock, BLOCKSIZE>>>(1, arrLength, braceBrackets, arr);
  cudaDeviceSynchronize();
  cudaFree(res);
  long* longRes;
  cudaMallocManaged(&longRes, arrLength*sizeof(long));
  thrust::inclusive_scan(thrust::cuda::par, arr, arr + arrLength, longRes);
  cudaDeviceSynchronize();
  end = clock();
  //std::cout << "-------------Curretness Second Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //print(longRes, arrLength, ROW1);
  //std::cout << "-------------End Second Step--------------" << std::endl;

  cudaFree(arr);
  start = clock();
  arr = countNodesRepitition(arrLength, numBlock, longRes);
  end = clock();
  //std::cout << "-------------Curretness Third Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //print(arr, arrLength, ROW1);
  //std::cout << "-------------End Third Step--------------" << std::endl;

  start = clock();
  checkCurrectenss<<<numBlock, BLOCKSIZE>>>(arrLength, braceBrackets, arr, longRes);
  cudaDeviceSynchronize();
  thrust::inclusive_scan(thrust::cuda::par, longRes, longRes + arrLength, longRes);
  cudaDeviceSynchronize();
  end = clock();
  //std::cout << "-------------Curretness Fourth Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //print(longRes, arrLength, ROW1);
  //std::cout << "-------------End Fourth Step--------------" << std::endl;

  allEnd = clock();
  //std::cout << "-------------isCorrect--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(allEnd-allStart)/CLOCKS_PER_SEC)*1000 << std::endl;
  //printf("%d\n", longRes[arrLength - 1] == arrLength);
  //std::cout << "-------------End isCorrect--------------" << std::endl;

  if(longRes[arrLength - 1] == arrLength) return true;
  return false;
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
  cudaMallocManaged(&res, length*ROW2*sizeof(long));
  cudaMallocManaged(&tmp, length*ROW1*sizeof(long));
  cudaMemcpy(res, arr,  length*ROW2*sizeof(long), cudaMemcpyHostToHost);
  tmp = sort(length, numBlock, arr);
  cudaMemcpy((res+length), tmp,  length*ROW1*sizeof(long), cudaMemcpyHostToHost);  
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
  int nextP2 = length == 1 ? 1 : 1 << (32 - __lzcnt(length-1));
  long * cudaArr;
  long * res;
  cudaMallocManaged(&cudaArr, length*ROW2*sizeof(long));
  cudaMallocManaged(&res, length*ROW2*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*sizeof(long), cudaMemcpyHostToDevice);
  int i = -1;
  for(int n = nextP2*2; n>1; n=n>>1){
    propagateParentsAndCountChildrenStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, res);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaArr, res,  length*ROW2*sizeof(long), cudaMemcpyDeviceToDevice);    
    i+=1;
  }
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
  for(long i=index; i<length; i+=stride){
    arr[length*ROW3 + i] = arr[length*ROW3 + i] + 1;
  }
}

long * allocate(int length, int numBlock, long* arr)
{
  long * cudaArr;
  cudaMallocManaged(&cudaArr, length*ROW4*sizeof(long));
  cudaMemcpy(cudaArr, arr,  length*ROW3*sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaArr+length*ROW3+1, arr+length*ROW2,  (length*ROW1-1)*sizeof(long), cudaMemcpyHostToDevice);
  cudaArr[length*ROW3] = 0;
  addOne<<<numBlock, BLOCKSIZE>>>(length, cudaArr);
  cudaDeviceSynchronize();
  cudaArr[length*ROW3] = 0;
  thrust::inclusive_scan(thrust::cuda::par, cudaArr+ROW3*length, cudaArr + ROW4*length, cudaArr+ROW3*length);
  cudaArr[length*ROW3] = 0;
  return cudaArr;
}

long * scan(int length, long* arr)
{
  long * cudaArr;
  long * res;
  cudaMallocManaged(&cudaArr, length*ROW4*sizeof(long));
  cudaMallocManaged(&res, length*ROW1*sizeof(long));
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
  allStart = clock();
  cudaMallocManaged(&attacheArr, attachedLength*sizeof(char));
  cudaMemcpy(attacheArr, input, length*sizeof(char), cudaMemcpyHostToDevice);
  attacheArr[length] = ',';
  char* sameDepthArr;
  cudaMallocManaged(&sameDepthArr, attachedLength*sizeof(char));
  //cudaMemset(sameDepthArr, 0, attachedLength*ROW1*sizeof(char));
  cudaMemcpy(sameDepthArr, attacheArr, attachedLength*sizeof(char), cudaMemcpyHostToDevice);
  //printString(attacheArr+attachedLength/100, attachedLength/100, ROW1);

  start = clock();
  changeDepth<<<numBlock, BLOCKSIZE>>>(attachedLength, attacheArr, sameDepthArr);
  cudaDeviceSynchronize();
  end = clock();
  cudaFree(attacheArr);
  //std::cout << "-------------First Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //printString(sameDepthArr, attachedLength, ROW1);
  //std::cout << "-------------End First Step--------------" << std::endl;

  start = clock();
  fakeRes = findDepthAndCount(attachedLength, numBlock, sameDepthArr);
  end = clock();
  //std::cout << "-------------Second Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //print(fakeRes, attachedLength, ROW3);
  //std::cout << "-------------End Second Step--------------" << std::endl;
  int arrLength = *(fakeRes+attachedLength-1);
  if(*fakeRes != -1){
    bool correct;
    correct = isCorrect(attachedLength, fakeRes+(attachedLength)*ROW2, sameDepthArr);
    if(correct){
      cudaMallocManaged(&arr, attachedLength*ROW4*sizeof(long));
      cudaMallocManaged(&res, arrLength*ROW4*sizeof(long));
      cudaMemcpy(arr, fakeRes,  attachedLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(fakeRes);
      start = clock();
      reduce<<<numBlock, BLOCKSIZE>>>(attachedLength, arrLength, sameDepthArr, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Third Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW2);
      //std::cout << "-------------End Third Step--------------" << std::endl;
      cudaFree(sameDepthArr);
      int numBlock = (arrLength + BLOCKSIZE - 1) / BLOCKSIZE;

      cudaMemcpy(arr, res,  arrLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      start = clock();
      fakeRes = sortByDepth(arrLength, numBlock, arr);
      end = clock();
      //std::cout << "-------------Fourth Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(fakeRes, arrLength, ROW2);
      //std::cout << "-------------End Fourth Step--------------" << std::endl;


      cudaMemcpy(arr, fakeRes,  arrLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(fakeRes);
      cudaMemset(res, -1, arrLength*ROW1*sizeof(long));
      start = clock();
      findParents<<<numBlock, BLOCKSIZE>>>( arrLength, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Fifth Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW1);
      //std::cout << "-------------End Fifth Step--------------" << std::endl;

      cudaMemcpy(arr, res,  arrLength*ROW1*sizeof(long), cudaMemcpyHostToHost);
      start = clock();
      fakeRes = propagateParentsAndCountChildren(arrLength, numBlock, arr);
      end = clock();
      //std::cout << "-------------Sixth Step--------------" << std::endl;      
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(fakeRes, arrLength, ROW2);
      //std::cout << "-------------End Sixth Step--------------" << std::endl;

      cudaMemcpy(arr, fakeRes,  arrLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(fakeRes);
      cudaMemset(res, -1, arrLength*ROW3*sizeof(long));
      start = clock();
      childsNumber<<<numBlock, BLOCKSIZE>>>(arrLength, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Seventh Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW3);
      //std::cout << "-------------End Seventh Step--------------" << std::endl;

      cudaMemcpy(arr, res,  arrLength*ROW3*sizeof(long), cudaMemcpyHostToHost);
      start = clock();
      fakeRes = allocate(arrLength, numBlock, arr);
      end = clock();
      //std::cout << "-------------Eighth Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(fakeRes, arrLength, ROW4);
      //std::cout << "-------------End Eighth Step--------------" << std::endl;

      long* sumRes;
      cudaMallocManaged(&sumRes, arrLength*ROW1*sizeof(long));
      start = clock();
      sumRes = scan(arrLength, fakeRes);
      end = clock();
      //std::cout << "-------------Scan Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(sumRes, arrLength, ROW1);
      //std::cout << "-------------End Scan Step--------------" << std::endl;
      long resLength = sumRes[arrLength - 1];
      cudaMemcpy(arr, fakeRes,  arrLength*ROW4*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(sumRes);
      cudaFree(fakeRes);
      cudaFree(res);
      cudaMallocManaged(&res, (arrLength+resLength)*sizeof(long));
      cudaMemset(res, 0, (arrLength+resLength)*sizeof(long));
      start = clock();
      generateRes<<<numBlock, BLOCKSIZE>>>(arrLength,  arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Last Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, (arrLength+resLength), ROW1);
      //std::cout << "-------------End Last Step--------------" << std::endl;
      allEnd = clock();
      cudaProfilerStop();     
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


int main(void)
{
  char ** names = (char**) malloc(FILESCOUNT*sizeof(char *));
  int * fileLength = (int *)malloc(FILESCOUNT*sizeof(int));
  int length;
  char ** texts;
  for(int i = 0; i< FILESCOUNT; i++){
    names[i] = (char *)malloc(NAMELENGTH*sizeof(char));
    strcpy(names[i], FILENAMES[i]);
  }
  int type;
  std::cout << "Please enter mode(single: 0, batch: 1): ";
  std::cin >> type;
  if(type){
    texts = loadMultipleFiles(FILESCOUNT, names, fileLength);
    double GPUparallelTime = NewRuntime_Parallel_GPU(texts[0], fileLength[0]);
    if(texts){
      for(int i=0; i<FILESCOUNT; i++){
        std::cout <<  fileLength[i] << std::endl;
        GPUparallelTime = NewRuntime_Parallel_GPU(texts[i], fileLength[i]);
        std::cout << "Parallel GPU time elapsed: " << std::setprecision (17) << (GPUparallelTime/CLOCKS_PER_SEC)*1000 << "ms." << std::endl;
      }
    }
  }
  else{
    char* input = loadFile(&length);
    if(input){
      std::cout <<  length << std::endl;
      double GPUparallelTime = NewRuntime_Parallel_GPU(input, length);
      GPUparallelTime = NewRuntime_Parallel_GPU(input, length);
      std::cout << "Parallel GPU time elapsed: " << std::setprecision (17) << (GPUparallelTime/CLOCKS_PER_SEC)*1000 << "ms." << std::endl;
    }
  }
  return 0;
}