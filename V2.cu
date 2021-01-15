
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
    if(step == 2){
      if(i > 0){
        int preChar = (int) strArr[i-1];
        if((currentChar == CLOSEBRACE || currentChar == CLOSEBRACKET) && !(preChar == CLOSEBRACKET || preChar == CLOSEBRACE)){
          res[i] = (long) i;
        }
        else if(currentChar == COMMA && (preChar != CLOSEBRACE && preChar != CLOSEBRACKET)){
          res[i] = -1;
        }
        else{
          res[i] = 0;
        }
      }
      else{
        res[i] = 0;
      }
    }
  }
}

__global__
void propagateIndexStep(int length, long* arr, int i, long* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long j = index; j< length; j+=stride){
    int pow2 = 1<<i;
    if(arr[j] != -1){
      if(j >= pow2){
        if(arr[j] < arr[j - pow2] && arr[j - pow2] != -1){
          res[j] = arr[j - pow2];
        }
        else{
          res[j] = arr[j];
        }
      }
      else{
        res[j] = arr[j];
      }
    }
    else{
      res[j] = -1;
    }
  }
}

long *propagateIndex(int length, int numBlock, char* strArr)
{
  int nextP2 = length == 1 ? 1 : 1 << (32 - __lzcnt(length-1));
  long * cudaArr;
  long * res;
  cudaMallocManaged(&cudaArr, length*ROW1*sizeof(long));
  cudaMallocManaged(&res, length*ROW1*sizeof(long));
  initialize<<<numBlock, BLOCKSIZE>>>(2, length, strArr, cudaArr);
  cudaDeviceSynchronize();
  int i = 0;
  for(int n = nextP2; n>1; n=n>>1){
    propagateIndexStep<<<numBlock, BLOCKSIZE>>>(length, cudaArr, i, res);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaArr, res,  length*ROW1*sizeof(long), cudaMemcpyDeviceToDevice);    
    i+=1;
  }
  return res;
 
}

__global__
void changeDepth(int length, char* strArr, long* arr, char* res)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(long i = index; i< length; i+=stride)
  {
    int currentChar = (int) strArr[i];
    if(currentChar == COMMA && arr[i] != -1){
      int preChar = (int)strArr[arr[i] - 1];
      if(preChar == OPENBRACE || preChar == OPENBRACKET){
        res[arr[i]] = I;
      }
      else{
        res[arr[i]] = strArr[i];
      }
    }
    else if(currentChar == CLOSEBRACKET || currentChar == CLOSEBRACE){
      res[i+1] = strArr[i];
    }
    else{
      res[i] = strArr[i];
    }
  }
}



double NewRuntime_Parallel_GPU(char* input, int length) {
  cudaProfilerStart();
  int attachedLength = length + 1;
  int numBlock = (attachedLength + BLOCKSIZE - 1) / BLOCKSIZE;
  long* res;
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
  //printString(attacheArr, attachedLength, ROW1);
  long * indexedRes;
  start = clock();
  indexedRes = propagateIndex(attachedLength, numBlock, attacheArr);
  changeDepth<<<numBlock, BLOCKSIZE>>>(attachedLength, attacheArr, indexedRes, sameDepthArr);
  cudaDeviceSynchronize();
  end = clock();
  cudaFree(attacheArr);
  //std::cout << "-------------First Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //printString(sameDepthArr, attachedLength, ROW1);
  //std::cout << "-------------End First Step--------------" << std::endl;

  start = clock();
  res = findDepthAndCount(attachedLength, numBlock, sameDepthArr);
  end = clock();
  //std::cout << "-------------Second Step--------------" << std::endl;
  //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  //print(res, attachedLength, ROW3);
  //std::cout << "-------------End Second Step--------------" << std::endl;
  int arrLength = *(res+attachedLength-1);
  if(*res != -1){
    bool correct;
    correct = isCorrect(attachedLength, res+(attachedLength)*ROW2, sameDepthArr);
    if(correct){
      char * cudaString;
      cudaMallocManaged(&cudaString, attachedLength*ROW1*sizeof(char));
      cudaMemcpy(cudaString, sameDepthArr,  attachedLength*sizeof(char), cudaMemcpyHostToDevice);
      cudaMallocManaged(&arr, attachedLength*ROW2*sizeof(long));
      cudaMemcpy(arr, res,  attachedLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(res);
      cudaMallocManaged(&res, arrLength*ROW2*sizeof(long));
      start = clock();
      reduce<<<numBlock, BLOCKSIZE>>>(attachedLength, arrLength, cudaString, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Third Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW2);
      //std::cout << "-------------End Third Step--------------" << std::endl;
      cudaFree(sameDepthArr);
      int numBlock = (arrLength + BLOCKSIZE - 1) / BLOCKSIZE;

      cudaFree(arr);
      cudaMallocManaged(&arr, arrLength*ROW2*sizeof(long));
      cudaMemcpy(arr, res,  arrLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(res);
      cudaMallocManaged(&res, arrLength*ROW2*sizeof(long));
      start = clock();
      res = sortByDepth(arrLength, numBlock, arr);
      end = clock();
      //std::cout << "-------------Fourth Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW2);
      //std::cout << "-------------End Fourth Step--------------" << std::endl;


      cudaFree(arr);
      cudaMallocManaged(&arr, arrLength*ROW2*sizeof(long));
      cudaMemcpy(arr, res,  arrLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(res);
      cudaMallocManaged(&res, arrLength*ROW1*sizeof(long));
      cudaMemset(res, -1, arrLength*ROW1*sizeof(long));
      start = clock();
      findParents<<<numBlock, BLOCKSIZE>>>( arrLength, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Fifth Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW1);
      //std::cout << "-------------End Fifth Step--------------" << std::endl;

      cudaFree(arr);
      cudaMallocManaged(&arr, arrLength*ROW1*sizeof(long));
      cudaMemcpy(arr, res,  arrLength*ROW1*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(res);
      cudaMallocManaged(&res, arrLength*ROW2*sizeof(long));
      start = clock();
      res = propagateParentsAndCountChildren(arrLength, numBlock, arr);
      end = clock();
      //std::cout << "-------------Sixth Step--------------" << std::endl;      
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW2);
      //std::cout << "-------------End Sixth Step--------------" << std::endl;


      cudaFree(arr);
      cudaMallocManaged(&arr, arrLength*ROW2*sizeof(long));
      cudaMemcpy(arr, res,  arrLength*ROW2*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(res);
      cudaMallocManaged(&res, arrLength*ROW3*sizeof(long));
      cudaMemset(res, -1, arrLength*ROW3*sizeof(long));
      start = clock();
      childsNumber<<<numBlock, BLOCKSIZE>>>(arrLength, arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Seventh Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW3);
      //std::cout << "-------------End Seventh Step--------------" << std::endl;

      cudaFree(arr);
      cudaMallocManaged(&arr, arrLength*ROW3*sizeof(long));
      cudaMemcpy(arr, res,  arrLength*ROW3*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(res);
      cudaMallocManaged(&res, arrLength*ROW4*sizeof(long));
      start = clock();
      res = allocate(arrLength, numBlock, arr);
      end = clock();
      //std::cout << "-------------Eighth Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, arrLength, ROW4);
      //std::cout << "-------------End Eighth Step--------------" << std::endl;

      long* sumRes;
      //cudaFree(arr);
      //cudaMallocManaged(&arr, arrLength*ROW4*sizeof(long));
      cudaMallocManaged(&sumRes, arrLength*ROW4*sizeof(long));
      cudaMemcpy(sumRes, res,  arrLength*ROW4*sizeof(long), cudaMemcpyHostToHost);
      start = clock();
      sumRes = scan(arrLength, sumRes);
      end = clock();
      //std::cout << "-------------Scan Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(sumRes, arrLength, ROW4);
      //std::cout << "-------------End Scan Step--------------" << std::endl;
      long resLength = sumRes[arrLength*ROW3 - 1];
      cudaFree(arr);
      cudaMallocManaged(&arr, arrLength*ROW4*sizeof(long));
      cudaMemcpy(arr, res,  arrLength*ROW4*sizeof(long), cudaMemcpyHostToHost);
      cudaFree(res);
      cudaMallocManaged(&res, (arrLength+resLength)*sizeof(long));
      start = clock();
      generateRes<<<numBlock, BLOCKSIZE>>>(arrLength,  arr, res);
      cudaDeviceSynchronize();
      end = clock();
      //std::cout << "-------------Last Step--------------" << std::endl;
      //std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
      //print(res, (arrLength+sumRes[arrLength*ROW3 - 1]), ROW1);
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
