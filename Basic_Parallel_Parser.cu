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

#define BLOCKSIZE 256
#define TOKENS 1000000

int print_d(char* input_d, int length){
  char * input;
  input = (char*) malloc(sizeof(char)*length);
  cudaMemcpy(input, input_d, sizeof(char)*length, cudaMemcpyDeviceToHost);
  printf("%s\n", input);
  free(input);
  return 1;
}


/*
 * MIT License
 *
 * Copyright (c) 2010 Serge Zaitsev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
 #ifndef JSMN_H
 #define JSMN_H
 
 #include <stddef.h>
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 #ifdef JSMN_STATIC
 #define JSMN_API static
 #else
 #define JSMN_API extern
 #endif
 
 /**
  * JSON type identifier. Basic types are:
  * 	o Object
  * 	o Array
  * 	o String
  * 	o Other primitive: number, boolean (true/false) or null
  */
 typedef enum {
   JSMN_UNDEFINED = 0,
   JSMN_OBJECT = 1 << 0,
   JSMN_ARRAY = 1 << 1,
   JSMN_STRING = 1 << 2,
   JSMN_PRIMITIVE = 1 << 3
 } jsmntype_t;
 
 enum jsmnerr {
   /* Not enough tokens were provided */
   JSMN_ERROR_NOMEM = -1,
   /* Invalid character inside JSON string */
   JSMN_ERROR_INVAL = -2,
   /* The string is not a full JSON packet, more bytes expected */
   JSMN_ERROR_PART = -3
 };
 
 /**
  * JSON token description.
  * type		type (object, array, string etc.)
  * start	start position in JSON data string
  * end		end position in JSON data string
  */
 typedef struct jsmntok {
   jsmntype_t type;
   int start;
   int end;
   int size;
 #ifdef JSMN_PARENT_LINKS
   int parent;
 #endif
 } jsmntok_t;
 
 /**
  * JSON parser. Contains an array of token blocks available. Also stores
  * the string being parsed now and current position in that string.
  */
 typedef struct jsmn_parser {
   unsigned int pos;     /* offset in the JSON string */
   unsigned int toknext; /* next token to allocate */
   int toksuper;         /* superior token node, e.g. parent object or array */
 } jsmn_parser;
 
 /**
  * Create JSON parser over an array of tokens
  */
  __global__
 JSMN_API void jsmn_init(jsmn_parser *parser, int length);
 
 /**
  * Run JSON parser. It parses a JSON data string into and array of tokens, each
  * describing
  * a single JSON object.
  */
  __global__
 JSMN_API void jsmn_parse(jsmn_parser *parser_s, const char **js_s, const size_t *len_s,
                         jsmntok_t *tokens_s, const unsigned int num_tokens_s, int total_size, int *error);
 
 #ifndef JSMN_HEADER
 /**
  * Allocates a fresh unused token from the token pool.
  */
__device__
 static jsmntok_t *jsmn_alloc_token(jsmn_parser *parser, jsmntok_t *tokens,
                                    const size_t num_tokens) {
   jsmntok_t *tok;
   if (parser->toknext >= num_tokens) {
     return NULL;
   }
   tok = &tokens[parser->toknext++];
   tok->start = tok->end = -1;
   tok->size = 0;
 #ifdef JSMN_PARENT_LINKS
   tok->parent = -1;
 #endif
   return tok;
 }
 
 /**
  * Fills token type and boundaries.
  */
  __device__
 static void jsmn_fill_token(jsmntok_t *token, const jsmntype_t type,
                             const int start, const int end) {
   token->type = type;
   token->start = start;
   token->end = end;
   token->size = 0;
 }
 
 /**
  * Fills next available token with JSON primitive.
  */
  __device__
 static int jsmn_parse_primitive(jsmn_parser *parser, const char *js,
                                 const size_t len, jsmntok_t *tokens,
                                 const size_t num_tokens) {
   jsmntok_t *token;
   int start;
 
   start = parser->pos;
 
   for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
     switch (js[parser->pos]) {
 #ifndef JSMN_STRICT
     /* In strict mode primitive must be followed by "," or "}" or "]" */
     case ':':
 #endif
     case '\t':
     case '\r':
     case '\n':
     case ' ':
     case ',':
     case ']':
     case '}':
       goto found;
     default:
                    /* to quiet a warning from gcc*/
       break;
     }
     if (js[parser->pos] < 32 || js[parser->pos] >= 127) {
       parser->pos = start;
       return JSMN_ERROR_INVAL;
     }
   }
 #ifdef JSMN_STRICT
   /* In strict mode primitive must be followed by a comma/object/array */
   parser->pos = start;
   return JSMN_ERROR_PART;
 #endif
 
 found:
   if (tokens == NULL) {
     parser->pos--;
     return 0;
   }
   token = jsmn_alloc_token(parser, tokens, num_tokens);
   if (token == NULL) {
     parser->pos = start;
     return JSMN_ERROR_NOMEM;
   }
   jsmn_fill_token(token, JSMN_PRIMITIVE, start, parser->pos);
 #ifdef JSMN_PARENT_LINKS
   token->parent = parser->toksuper;
 #endif
   parser->pos--;
   return 0;
 }
 
 /**
  * Fills next token with JSON string.
  */
  __device__
 static int jsmn_parse_string(jsmn_parser *parser, const char *js,
                              const size_t len, jsmntok_t *tokens,
                              const size_t num_tokens) {
   jsmntok_t *token;
 
   int start = parser->pos;
   
   /* Skip starting quote */
   parser->pos++;
   
   for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
     char c = js[parser->pos];
 
     /* Quote: end of string */
     if (c == '\"') {
       if (tokens == NULL) {
         return 0;
       }
       token = jsmn_alloc_token(parser, tokens, num_tokens);
       if (token == NULL) {
         parser->pos = start;
         return JSMN_ERROR_NOMEM;
       }
       jsmn_fill_token(token, JSMN_STRING, start + 1, parser->pos);
 #ifdef JSMN_PARENT_LINKS
       token->parent = parser->toksuper;
 #endif
       return 0;
     }
 
     /* Backslash: Quoted symbol expected */
     if (c == '\\' && parser->pos + 1 < len) {
       int i;
       parser->pos++;
       switch (js[parser->pos]) {
       /* Allowed escaped symbols */
       case '\"':
       case '/':
       case '\\':
       case 'b':
       case 'f':
       case 'r':
       case 'n':
       case 't':
         break;
       /* Allows escaped symbol \uXXXX */
       case 'u':
         parser->pos++;
         for (i = 0; i < 4 && parser->pos < len && js[parser->pos] != '\0';
              i++) {
           /* If it isn't a hex character we have an error */
           if (!((js[parser->pos] >= 48 && js[parser->pos] <= 57) ||   /* 0-9 */
                 (js[parser->pos] >= 65 && js[parser->pos] <= 70) ||   /* A-F */
                 (js[parser->pos] >= 97 && js[parser->pos] <= 102))) { /* a-f */
             parser->pos = start;
             return JSMN_ERROR_INVAL;
           }
           parser->pos++;
         }
         parser->pos--;
         break;
       /* Unexpected symbol */
       default:
         parser->pos = start;
         return JSMN_ERROR_INVAL;
       }
     }
   }
   parser->pos = start;
   return JSMN_ERROR_PART;
 }

 /**
  * Parse JSON string and fill tokens.
  */
  __global__
 JSMN_API void jsmn_parse(jsmn_parser *parser_s, const char **js_s, const size_t *len_s,
                         jsmntok_t *tokens_s, const unsigned int num_tokens, int total_size, int *error) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("%d\n", 110);
    for(int k=index; k<total_size; k+=stride){
      jsmn_parser *parser = parser_s+k;
      const char * js = js_s[k];
      const size_t len = len_s[k];
      jsmntok_t *tokens = tokens_s+(k*num_tokens);
    int r;
   int i;
   jsmntok_t *token;
   int count = parser->toknext;
 
   for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
     char c;
     jsmntype_t type;

     c = js[parser->pos];
     switch (c) {
     case '{':
     case '[':
       count++;
       if (tokens == NULL) {
         break;
       }
       token = jsmn_alloc_token(parser, tokens, num_tokens);
       if (token == NULL) {
         error[k] = JSMN_ERROR_NOMEM;
         return ; //JSMN_ERROR_NOMEM;
       }
       if (parser->toksuper != -1) {
         jsmntok_t *t = &tokens[parser->toksuper];
 #ifdef JSMN_STRICT
         /* In strict mode an object or array can't become a key */
         if (t->type == JSMN_OBJECT) {
           error[k] = JSMN_ERROR_INVAL;
           return ;//JSMN_ERROR_INVAL;
         }
 #endif
         t->size++;
 #ifdef JSMN_PARENT_LINKS
         token->parent = parser->toksuper;
 #endif
       }
       token->type = (c == '{' ? JSMN_OBJECT : JSMN_ARRAY);
       token->start = parser->pos;
       parser->toksuper = parser->toknext - 1;
       break;
     case '}':
     case ']':
       if (tokens == NULL) {
         break;
       }
       type = (c == '}' ? JSMN_OBJECT : JSMN_ARRAY);
 #ifdef JSMN_PARENT_LINKS
       if (parser->toknext < 1) {
          error[k] = JSMN_ERROR_INVAL;
         return ;//JSMN_ERROR_INVAL;
       }
       token = &tokens[parser->toknext - 1];
       for (;;) {
         if (token->start != -1 && token->end == -1) {
           if (token->type != type) {
             error[k] = JSMN_ERROR_INVAL;
             return ;//JSMN_ERROR_INVAL;
           }
           token->end = parser->pos + 1;
           parser->toksuper = token->parent;
           break;
         }
         if (token->parent == -1) {
           if (token->type != type || parser->toksuper == -1) {
             error[k] = JSMN_ERROR_INVAL;
             return ;//JSMN_ERROR_INVAL;
           }
           break;
         }
         token = &tokens[token->parent];
       }
 #else
       for (i = parser->toknext - 1; i >= 0; i--) {
         token = &tokens[i];
         if (token->start != -1 && token->end == -1) {
           if (token->type != type) {
             error[k] = JSMN_ERROR_INVAL;
             return; // JSMN_ERROR_INVAL;
           }
           parser->toksuper = -1;
           token->end = parser->pos + 1;
           break;
         }
       }
       /* Error if unmatched closing bracket */
       if (i == -1) {
         error[k] = JSMN_ERROR_INVAL;
         return ;//JSMN_ERROR_INVAL;
       }
       for (; i >= 0; i--) {
         token = &tokens[i];
         if (token->start != -1 && token->end == -1) {
           parser->toksuper = i;
           break;
         }
       }
 #endif
       break;
     case '\"':
       r = jsmn_parse_string(parser, js, len, tokens, num_tokens);
       if (r < 0) {
         error[k] = r;
         return ;//r;
       }
       count++;
       if (parser->toksuper != -1 && tokens != NULL) {
         tokens[parser->toksuper].size++;
       }
       break;
     case '\t':
     case '\r':
     case '\n':
     case ' ':
       break;
     case ':':
       parser->toksuper = parser->toknext - 1;
       break;
     case ',':
       if (tokens != NULL && parser->toksuper != -1 &&
           tokens[parser->toksuper].type != JSMN_ARRAY &&
           tokens[parser->toksuper].type != JSMN_OBJECT) {
 #ifdef JSMN_PARENT_LINKS
         parser->toksuper = tokens[parser->toksuper].parent;
 #else
         for (i = parser->toknext - 1; i >= 0; i--) {
           if (tokens[i].type == JSMN_ARRAY || tokens[i].type == JSMN_OBJECT) {
             if (tokens[i].start != -1 && tokens[i].end == -1) {
               parser->toksuper = i;
               break;
             }
           }
         }
 #endif
       }
       break;
 #ifdef JSMN_STRICT
     /* In strict mode primitives are: numbers and booleans */
     case '-':
     case '0':
     case '1':
     case '2':
     case '3':
     case '4':
     case '5':
     case '6':
     case '7':
     case '8':
     case '9':
     case 't':
     case 'f':
     case 'n':
       /* And they must not be keys of the object */
       if (tokens != NULL && parser->toksuper != -1) {
         const jsmntok_t *t = &tokens[parser->toksuper];
         if (t->type == JSMN_OBJECT ||
             (t->type == JSMN_STRING && t->size != 0)) {
               error[k] = JSMN_ERROR_INVAL;
           return ;//JSMN_ERROR_INVAL;
         }
       }
 #else
     /* In non-strict mode every unquoted value is a primitive */
     default:
 #endif
       r = jsmn_parse_primitive(parser, js, len, tokens, num_tokens);
       if (r < 0) {
         error[k] = r;
         return ;//r;
       }
       count++;
       if (parser->toksuper != -1 && tokens != NULL) {
         tokens[parser->toksuper].size++;
       }
       break;
 
 #ifdef JSMN_STRICT
     /* Unexpected char in strict mode */
     default:
       error[k] = JSMN_ERROR_INVAL;
       return; //JSMN_ERROR_INVAL;
 #endif
     }
   }

   if (tokens != NULL) {
     for (i = parser->toknext - 1; i >= 0; i--) {
       /* Unmatched opened object or array */
       if (tokens[i].start != -1 && tokens[i].end == -1) {
         error[k] = JSMN_ERROR_PART;
         return ;//JSMN_ERROR_PART;
       }
     }
   }
   error[k] = count;
   return ;//count;
  }
 }
 
 /**
  * Creates a new parser based over a given buffer with an array of tokens
  * available.
  */
  __global__
 JSMN_API void jsmn_init(jsmn_parser *parser, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i<length; i+=stride){
      (parser+i)->pos = 0;
      (parser+i)->toknext = 0;
      (parser+i)->toksuper = -1;

    }
 }
 
 #endif /* JSMN_HEADER */
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* JSMN_H */


void host_to_device(char ** js, size_t * sizes, const char ** &js_d, size_t * &sizes_d, int length){
  char ** middle;
  middle = (char **)malloc(sizeof(char*)*length);
  char ** back_d;
  for(int i = 0; i<length; i++){
    cudaMalloc(middle+i, sizeof(char)*sizes[i]);
    cudaMemcpy(middle[i], js[i], sizeof(char)*sizes[i], cudaMemcpyHostToDevice);
  }
  cudaMalloc(&sizes_d, sizeof(size_t)*length);
  cudaMemcpy(sizes_d, sizes, sizeof(size_t)*length, cudaMemcpyHostToDevice);
  cudaMalloc(&js_d, sizeof(char *)*length);
  cudaMemcpy(js_d, middle, sizeof(char *)*length, cudaMemcpyHostToDevice);
  //js_d = back_d;

}

uint64_t file_read(char ** &records, size_t * &records_size, int &total_size){
  FILE *fp;
  fp=fopen("./../Large-Json/walmart_small_records.json", "r");
  long int lines =0;

  if ( fp == NULL ) {
    return 0;
  }

  while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
        ++lines;

  total_size = lines;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  rewind(fp);

  char ** total_lines = (char **)malloc(sizeof(char *)* lines);
  size_t * lines_size = (size_t *)malloc(sizeof(size_t)*lines);

  records = (char **)malloc(sizeof(char *)* lines);
  records_size = (size_t *)malloc(sizeof(size_t)*lines);

  if (fp == NULL)
      exit(EXIT_FAILURE);

  int i=0;
  uint64_t read_byte_size = 0;
  while ((read = getline(&line, &len, fp)) != -1) {
      read_byte_size += read;
      //printf("Retrieved line of length %zu:\n", read);
      //printf("%s", line);
      records[i] = (char *)malloc(sizeof(char)* read);
      memcpy(records[i], line, sizeof(char)*read);
      records_size[i] = (size_t)read;
      i++;

  }

  //if (line) free(line);

  //records = total_lines;
  //memcpy(records, total_lines, sizeof(char *)*lines);
  //memcpy(records_size, lines_size, sizeof(size_t)*lines);

  //records_size = lines_size;
  
  fclose(fp);

  return read_byte_size;
}

int main(int argc, char **argv)
{
  //double result;

  clock_t start, end;

  char **js;
  size_t * sizes;
  int total_size;

  uint64_t read_byte_size = file_read(js, sizes, total_size);
  double line_average_size = read_byte_size/total_size;
  printf("line average: %f\n", line_average_size);
  uint32_t lines_to_parse = 33554432/line_average_size;
  uint32_t line_absolute_size = ((uint32_t) line_average_size << 1) >> 1;
  //32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 33554432 134217728 536870912 1073741824
  //exit(0);
  const char **js_d;
  size_t * sizes_d;

  int numBlock = (lines_to_parse + BLOCKSIZE - 1) / BLOCKSIZE;

  host_to_device(js, sizes, js_d, sizes_d, total_size);

  int total_allowed = lines_to_parse;
  int total_count = 0;
  double total_time = 0;
  while(total_count < total_size){
    int current_total = total_size - total_count > lines_to_parse ? total_allowed : total_size - total_count;

    double iteration_time = 0;
    //char ** temp = (char **)malloc(sizeof(char*)*current_total);
    //cudaMemcpy(temp, js_d, sizeof(char*)*current_total, cudaMemcpyDeviceToHost);
  
  
    //jsmn_parser *p;
    jsmn_parser *p_d;
    jsmn_parser *p = (jsmn_parser *)malloc(sizeof(jsmn_parser)*current_total);
  
    cudaMalloc(&p_d, sizeof(jsmn_parser)*current_total);
    jsmntok_t *t; // We expect no more than 128 JSON tokens
    jsmntok_t *t_h = (jsmntok_t *)malloc(sizeof(jsmntok_t)*(current_total*line_absolute_size));

    //printf("first character: %c\n", *(js[current_total-2]));


    start = clock();
    cudaMalloc(&t, (current_total*line_absolute_size)*sizeof(jsmntok_t));
    jsmn_init<<<numBlock, BLOCKSIZE>>>(p_d, (size_t)current_total);
    cudaDeviceSynchronize();
  
    //print_d(*(temp), sizes[0]);
    int *error;
    cudaMalloc(&error, sizeof(int)*current_total);
    jsmn_parse<<<numBlock, BLOCKSIZE>>>(p_d, js_d+total_count, sizes_d+total_count, t, line_absolute_size, current_total, error);
    cudaDeviceSynchronize();
    end = clock();
    iteration_time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    total_time += iteration_time;
    std::cout << "Time elapsed: " << std::setprecision (17) << iteration_time << std::endl;
  
    int *error_h = (int*)malloc(sizeof(int)*current_total);
    cudaMemcpy(error_h, error, sizeof(int)*current_total, cudaMemcpyDeviceToHost);
    cudaMemcpy(t_h, t, sizeof(jsmntok_t)*current_total*line_absolute_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(p, p_d, sizeof(jsmn_parser)*current_total, cudaMemcpyDeviceToHost);
    //printf("%.*s\n", t_h[(current_total-2)*line_absolute_size+5].end - t_h[(current_total-2)*line_absolute_size+5].start,
    //        js[current_total-2] + t_h[(current_total-2)*line_absolute_size+5].start);
    //for(int i=0; i<total_size; i++) printf("%d\n", error_h[i]);
    free(error_h);
    free(t_h);
    free(p);
    cudaFree(p_d);
    cudaFree(t);
    cudaFree(error);


    total_count += current_total;
    //break;
  }

  /*
  char ** temp = (char **)malloc(sizeof(char*)*total_size);
  cudaMemcpy(temp, js_d, sizeof(char*)*total_size, cudaMemcpyDeviceToHost);


  //jsmn_parser *p;
  jsmn_parser *p_d;
  jsmn_parser *p = (jsmn_parser *)malloc(sizeof(jsmn_parser)*total_size);

  cudaMalloc(&p_d, sizeof(jsmn_parser)*total_size);
  jsmntok_t *t; // We expect no more than 128 JSON tokens
  jsmntok_t *t_h = (jsmntok_t *)malloc(sizeof(jsmntok_t)*(total_size*10000));

  start = clock();
  cudaMalloc(&t, (total_size*10000)*sizeof(jsmntok_t));
  jsmn_init<<<numBlock, BLOCKSIZE>>>(p_d, (size_t)total_size);
  cudaDeviceSynchronize();

  //print_d(*(temp), sizes[0]);
  int *error;
  cudaMalloc(&error, sizeof(int)*total_size);
  jsmn_parse<<<numBlock, BLOCKSIZE>>>(p_d, js_d, sizes_d, t, 10000, total_size, error);
  cudaDeviceSynchronize();
  end = clock();
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

  int *error_h = (int*)malloc(sizeof(int)*total_size);
  cudaMemcpy(error_h, error, sizeof(int)*total_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(t_h, t, sizeof(jsmntok_t)*total_size*10000, cudaMemcpyDeviceToHost);
  cudaMemcpy(p, p_d, sizeof(jsmn_parser)*total_size, cudaMemcpyDeviceToHost);
  printf("%.*s\n", t_h[(total_size-1)*10000+2].end - t_h[(total_size-1)*10000+2].start, js[total_size-1] + t_h[(total_size-1)*10000+2].start);
  //for(int i=0; i<total_size; i++) printf("%d\n", error_h[i]);
  */
  printf("----------------------------------\n total time: %f \n-------------------------------\n", total_time);
  return 0;
}