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

#define BLOCKSIZE 32
#define TOKENS 1000000
#define THREADID 0
#define AVGGPUCLOCK 1346000000

int print_d(const char** input_d, int length){
  char * input;
  char ** input_ptr = (char **)malloc(sizeof(char*));
  cudaMemcpy(input_ptr, input_d, sizeof(char *), cudaMemcpyDeviceToHost);
  input = (char*) malloc(sizeof(char)*length);
  cudaMemcpy(input, *input_ptr, sizeof(char)*length, cudaMemcpyDeviceToHost);
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

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   int k = index;
 
   int start = parser->pos;

   int case_escaped = 0;
   int case_utf = 0;
   int total_case = 0;
   
   /* Skip starting quote */
   parser->pos++;
   
   for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
     char c = js[parser->pos];

     /* Quote: end of string */
     if (c == '\"') {
      //if(k==0 && parser->pos + 1 < len)
        //printf("position: %d, previous character: %c, current character: %c, next character: %c\n", parser->pos, js[(parser->pos)-1], js[(parser->pos)], js[(parser->pos)+1]);
       if (tokens == NULL) {
        // if(threadIdx.x == THREADID && blockIdx.x == 0) printf("%f\n",
        //                            ((double)case_escaped));
        // if(threadIdx.x == THREADID && blockIdx.x == 0) printf("%f\n",
        //                            ((double)(total_case-case_escaped)));
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
      //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("%f\n",
      //                             ((double)case_escaped));
      //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("%f\n",
      //                             ((double)(total_case-case_escaped)));
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
         if(threadIdx.x == THREADID && blockIdx.x == 0) case_escaped++;
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
         if(threadIdx.x == THREADID && blockIdx.x == 0) case_utf++;
         break;
       /* Unexpected symbol */
       default:
         parser->pos = start;
         return JSMN_ERROR_INVAL;
       }
     }
     if( threadIdx.x == THREADID && blockIdx.x == 0) total_case++;
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
    clock_t start, end;
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

   double thread_runtime = 0;
   double thread_string_runtime = 0;
   double thread_comma_runtime = 0;
   double thread_start_obj_runtime = 0;
   double thread_primitive_runtime = 0;

   int case_open = 0;
   int case_close = 0;
   int case_string = 0;
   int case_whitespace = 0;
   int case_colon = 0;
   int case_comma = 0;
   int case_primitive = 0;

   int current_state = 0;
   int state_changed_num = 0;

 
   for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
     char c;
     jsmntype_t type;
   //if(threadIdx.x == THREADID) printf("blockId: %d, position: %d\n", blockIdx.x, parser->pos);

     c = js[parser->pos];
     switch (c) {
     case '{': // if open brace/bracket allocate a token and if its a key return an error, otherwise set the type and starting point, check next char
     case '[':
     if(threadIdx.x == THREADID) case_open++;
     if(threadIdx.x == THREADID) start = clock();

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
       if(threadIdx.x == THREADID) end = clock();
       if(threadIdx.x == THREADID) thread_start_obj_runtime += ((double)(end-start)/AVGGPUCLOCK)*1000;
       break;
     case '}': // if it's end of an object, try to find the parent node for the current obj and set it as token->parent if the flag is set,
     case ']': // Also find the last token that doesn't have an ending point, using parent connection.
       if(threadIdx.x == THREADID) case_close++;
       if (tokens == NULL) {
         break;
       }
       type = (c == '}' ? JSMN_OBJECT : JSMN_ARRAY);
 #ifdef JSMN_PARENT_LINKS
       if(threadIdx.x == THREADID) start = clock();
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
       if(threadIdx.x == THREADID) end = clock();
       if(threadIdx.x == THREADID) thread_runtime += ((double)(end-start)/AVGGPUCLOCK)*1000;
       //if(threadIdx.x == THREADID && blockIdx.x == 45) printf("blockId: %d, %f\n", blockIdx.x, ((double)(end)/CLOCKS_PER_SEC)*1000 );
 #else //if flag is not set, find the last token with no ending using iteration on parsed tokens and set the current character as its ending token
       if(threadIdx.x == THREADID) start = clock();
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
       //if(threadIdx.x == 1) printf("blockId: %d\n", blockIdx.x);
       /* Error if unmatched closing bracket */
       if (i == -1) {
         error[k] = JSMN_ERROR_INVAL;
         return ;//JSMN_ERROR_INVAL;
       }
       for (; i >= 0; i--) { //find the next last object that doesn't have an ending and set that token as the parser's current parent.
         token = &tokens[i];
         if (token->start != -1 && token->end == -1) {
           parser->toksuper = i;
           break;
         }
       }
       if(threadIdx.x == THREADID) end = clock();
       if(threadIdx.x == THREADID) thread_runtime += ((double)(end-start)/AVGGPUCLOCK)*1000;
       //if(threadIdx.x == THREADID) printf("blockId: %d, %f\n", blockIdx.x, ((double)(end-start)/CLOCKS_PER_SEC)*1000 );

 #endif
       break;
     case '\"': // it's start of an string. call parse_string
       if(threadIdx.x == THREADID) case_string++;
       if(threadIdx.x == THREADID) start = clock();
       r = jsmn_parse_string(parser, js, len, tokens, num_tokens);
       if (r < 0) {
         error[k] = r;
         return ;//r;
       }
       count++;
       if (parser->toksuper != -1 && tokens != NULL) {
         tokens[parser->toksuper].size++;
       }
       if(threadIdx.x == THREADID) end = clock();
       if(threadIdx.x == THREADID) thread_string_runtime += ((double)(end-start)/AVGGPUCLOCK)*1000;
       break;
     case '\t':  // white space, just pass
     case '\r':
     case '\n':
     case ' ':
      if(threadIdx.x == THREADID) case_whitespace++;
       break;
     case ':': // update the parent token to the previouse token.
        if(threadIdx.x == THREADID) case_colon++;
       parser->toksuper = parser->toknext - 1;
       break;
     case ',': // this token is complete find the next one and its parent (use loops if we don't have parent links).
     if(threadIdx.x == THREADID) case_comma++;
     if(threadIdx.x == THREADID) start = clock();
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
       if(threadIdx.x == THREADID) end = clock();
       if(threadIdx.x == THREADID) thread_comma_runtime += ((double)(end-start)/AVGGPUCLOCK)*1000;
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
      if(threadIdx.x == THREADID) case_primitive++;
       if(threadIdx.x == THREADID) start = clock();
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
       if(threadIdx.x == THREADID) end = clock();
       if(threadIdx.x == THREADID) thread_primitive_runtime += ((double)(end-start)/AVGGPUCLOCK)*1000;
       break;
 
 #ifdef JSMN_STRICT
     /* Unexpected char in strict mode */
     default:
       error[k] = JSMN_ERROR_INVAL;
       return; //JSMN_ERROR_INVAL;
 #endif
     }
   }
  //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("thread total time: %f\n",
  //                              thread_start_obj_runtime+thread_string_runtime+thread_comma_runtime+thread_primitive_runtime);
  //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("thread start obj time: %f\n",thread_start_obj_runtime);
  //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("thread end obj time: %f\n",thread_runtime);
  //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("thread string time: %f\n",thread_string_runtime);
  //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("thread comma time: %f\n",thread_comma_runtime);
  //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("thread primitive time: %f\n",thread_primitive_runtime);

  // int total_case = case_open+case_close+case_string+case_whitespace+case_colon+case_comma+case_primitive;
  //  if(threadIdx.x == THREADID && blockIdx.x == 0) printf("thread start %f, end %f, string %f, whitespace %f, colon %f, comma %f, primitive %f\n",
  //                                    ((double)case_open),
  //                                    ((double)case_close),
  //                                    ((double)case_string),
  //                                    ((double)case_whitespace),
  //                                    ((double)case_colon),
  //                                    ((double)case_comma),
  //                                    ((double)case_primitive));
   

   if (tokens != NULL) { // if there is any remaining token that doesn't have and end point, return error
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


void total_free(const char ** &js_d, size_t * &sizes_d, int length, int lines_to_parse){
  char ** js = (char **)malloc(sizeof(char *)*lines_to_parse);
  cudaMemcpy(js, js_d, sizeof(char *)*lines_to_parse, cudaMemcpyDeviceToHost);
  for(int i=0; i<lines_to_parse; i++){
    //printf("fffffffff\n");
    cudaFree(js[i]);
  }
  cudaFree(js_d);
  free(js);
  cudaFree(sizes_d);


}

void host_to_device(char ** js, size_t * sizes, const char ** &js_d, size_t * &sizes_d, int length, int lines_to_parse, int iter){
  char ** middle;
  middle = (char **)malloc(sizeof(char*)*lines_to_parse);
  //char ** back_d;
  for(int i = 0; i<lines_to_parse && lines_to_parse*iter+i<length; i++){
    cudaMalloc(middle+i, sizeof(char)*sizes[lines_to_parse*iter+i]);
    //if(iter > 15 && i==55) printf("%d\n", (int)sizes[lines_to_parse*iter+i]);
    cudaMemcpy(middle[i], js[lines_to_parse*iter+i], sizeof(char)*sizes[lines_to_parse*iter+i], cudaMemcpyHostToDevice);
    //if(i==1) printf("%s\n", js[lines_to_parse*iter+i]);
  }
  cudaMalloc(&sizes_d, sizeof(size_t)*lines_to_parse);
  //printf("%d , %d\n", length, lines_to_parse*iter);
  if (lines_to_parse*(iter+1)>length) cudaMemcpy(sizes_d, sizes+(lines_to_parse*iter), sizeof(size_t)*(length - lines_to_parse*iter), cudaMemcpyHostToDevice);
  else cudaMemcpy(sizes_d, sizes+lines_to_parse*iter, sizeof(size_t)*lines_to_parse, cudaMemcpyHostToDevice);
  cudaMalloc(&js_d, sizeof(char *)*lines_to_parse);
  cudaMemcpy(js_d, middle, sizeof(char *)*lines_to_parse, cudaMemcpyHostToDevice);
  free(middle);
  //js_d = back_d;

}

uint64_t file_read(char ** &records, size_t * &records_size, int &total_size){
  FILE *fp;
  fp=fopen("./../Large-Json/wiki_small_records.json", "r"); // ./inputs/All_purpose_1000.txt ./../Large-Json/wiki_small_records.json
  long int lines = 0;

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
      *(records[i]+read-1) = 0;
      records_size[i] = (size_t)(read);
      //if(i==1) {printf("%s\n", records[i]); printf("%ld\n", records_size[i]);}
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

  start = clock();
  uint64_t read_byte_size = file_read(js, sizes, total_size);
  end = clock();
  std::cout << "Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

  //printf("%s\n", js[1]);
  //printf("%ld\n", sizes[1]);
  printf("total lines of file: %d\n", total_size);
  double line_average_size = read_byte_size/total_size;
  printf("line average: %f\n", line_average_size);
  uint32_t lines_to_parse = 33554432/line_average_size;
  //lines_to_parse = lines_to_parse > 20000 ? 20000 : lines_to_parse; 
  uint32_t line_absolute_size = ((uint32_t) line_average_size << 1) >> 2;
  printf("line absolute size: %d\n", line_absolute_size);
  //32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 33554432 134217728 536870912 1073741824
  //exit(0);

  //lines_to_parse = 10;
  int numBlock = (lines_to_parse + BLOCKSIZE - 1) / BLOCKSIZE;
  printf("Blocks: %d\n", numBlock);
  printf("size of tokens: %ld\n", sizeof(jsmntok_t));

  int total_allowed = lines_to_parse;
  int total_count = 0;
  double total_time = 0;

  int counter = 0;
  while(total_count < total_size){
    const char **js_d;
    size_t * sizes_d;

    start = clock();
    host_to_device(js, sizes, js_d, sizes_d, total_size, lines_to_parse, counter);
    end = clock();
    std::cout << "Copy Time elapsed: " << std::setprecision (17) << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;
  

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
    cudaMemset(error, sizeof(int)*current_total, 1);
      //*******************************//
      // size_t l_free = 0;
      // size_t l_Total = 0;
      // cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);
      // size_t allocated = (l_Total - l_free);
      // std::cout << "Total: " << l_Total << " Free: " << l_free << " Allocated: " << allocated << std::endl;
      //*******************************//
    //printf("%c\n", *((char*)*(js_d+total_count)));
    jsmn_parse<<<numBlock, BLOCKSIZE>>>(p_d, js_d, sizes_d, t, line_absolute_size, current_total, error);

    cudaDeviceSynchronize();

    end = clock();
  


    iteration_time = ((double)(end-start)/CLOCKS_PER_SEC)*1000;
    total_time += iteration_time;
    //printf("iteration end: %f\n", ((double)(end)/CLOCKS_PER_SEC)*1000);
    std::cout << "Time elapsed: " << std::setprecision (17) << iteration_time << std::endl;
  
    int *error_h = (int*)malloc(sizeof(int)*current_total);
    cudaMemcpy(error_h, error, sizeof(int)*current_total, cudaMemcpyDeviceToHost);
    cudaMemcpy(t_h, t, sizeof(jsmntok_t)*current_total*line_absolute_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(p, p_d, sizeof(jsmn_parser)*current_total, cudaMemcpyDeviceToHost);
    const int line_number = current_total;
    const int token_number = 4;
    // #ifdef JSMN_PARENT_LINKS
    // printf("Parent index: %d\n", t_h[(current_total-line_number)*line_absolute_size+token_number].parent);
    // int index_parent = (current_total-line_number)*line_absolute_size+t_h[(current_total-line_number)*line_absolute_size+token_number].parent;
    // printf("%.*s\n", t_h[index_parent].end - t_h[index_parent].start,
    //        js[total_count+current_total-line_number] + t_h[index_parent].start);
    // #endif
    //printf("%.*s\n", t_h[(current_total-line_number)*line_absolute_size+token_number].end - t_h[(current_total-line_number)*line_absolute_size+token_number].start,
    //        js[total_count+current_total-line_number] + t_h[(current_total-line_number)*line_absolute_size+token_number].start);
    //for(int i=0; i<current_total/100; i++) printf("%d\n", error_h[i]);
    free(error_h);
    free(t_h);
    free(p);
    cudaFree(p_d);
    cudaFree(t);
    cudaFree(error);

    //printf("%s\n", js[current_total-line_number]);
    //if(counter == 17) break;
    counter++;
    total_count += current_total;
    total_free(js_d, sizes_d, total_size, lines_to_parse);

  }

  printf("total_count: %d\n", total_count);

  //total_free(js_d, sizes_d, total_size);

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