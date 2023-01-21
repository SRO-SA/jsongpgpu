#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <x86intrin.h>
#include <string>
#include <vector>
#include <queue>
#include <inttypes.h>
#include <sys/resource.h>
#include <stdint.h>
#include <algorithm>
#include "Parse_Query.hpp"


using namespace std;

// enum tokens_type_enum { 
//     OBJECT,
//     ARRAY,
//     PRIMITIVE
// }; 

// enum primitive_type_enum { 
//     NUMBER,
//     TRUE,
//     FALSE,
//     NULL_TYPE,
//     STRING
// }; 

// typedef tokens_type_enum token_type;
// typedef primitive_type_enum primitive_type;


void structural_iterator::reset(){
    current_array_index = 0;
    parent_array_index = 0;
}

int structural_iterator::goto_index(int i){
    int index = i;
    if(buf_tree_index[current_array_index] < index){
        printf("error on setting new index\n");
        return 0;
    }
    parent_array_index = current_array_index;
    int child_index = buf_tree_index[current_array_index+index];
    current_array_index = child_index;
    //printf("parent is: %d and current node is %d\n", parent_array_index, current_array_index);
    return current_array_index;
}

string structural_iterator::getString(int start_index, int end_index){
    int i = 0;
    int length = 0;
    int state = 0;
    int error = 0;
    int string_start = 0;
    int string_end = 0;
    string result;
    char current_char = input_json[end_index];
    current_char == ':' ? end_index-- : NULL;
    current_char = input_json[end_index];
    while(current_char == ' '){
        current_char = input_json[--end_index];
    }
    length = end_index - start_index-1;
    result.assign((char*)(input_json+start_index+1), abs(length));
    return result;

    // while(i>=0){
    //     char current_char = input_json[start_index+i];
    //     if(state == 0){
    //         //printf("State 0: %d char: %c\n", i, input_json[start_index+i]);
    //         if(current_char == '"') {state = 1; string_start = i+1;}
    //         else if(current_char != '\t' && current_char != '\n' && current_char != '\r' && current_char != ' ') {error = 1; break;}
    //         i++;
    //     }
    //     else if(state == 1){
    //         //printf("State 1: %d char: %c\n", i, input_json[start_index+i]);
    //         if(current_char == '\\'){
    //             length+=2;
    //             i+=2;
    //             continue;
    //         }
    //         else if(current_char =='"'){
    //             state = 2;
    //             string_end = i-1;
    //         }
    //         length++;
    //         i++;
    //     }
    //     else if(state == 2){
    //         //printf("State 2: %d char: %c\n", i, input_json[start_index+i]);
    //         if (error != 1){
    //             result = string((char*)(input_json+start_index+string_start), abs(string_end-string_start)+1);
    //             //result+='\0';
    //             return result;
    //         }
    //         printf("There is an error in string at index: %d char: %c\n", i, input_json[start_index+i]);
    //         break;
    //     }

    // }
    if(error == 1 || result.empty()) printf("Couldn't get the string\n");
    return result;
}

string structural_iterator::getStringBackward(int end_index){
    int i = 1;
    int length = 0;
    int state = 0;
    int error = 0;
    int string_start = 0;
    int string_end = 0;
    string result;
    while(i>=0){
        char current_char = input_json[end_index-i];
        if(state == 0){
            //printf("State 0: %d char: %c\n", i, input_json[end_index-i]);
            if(current_char == '\"') {state = 1; string_end = i+1;}
            else if(current_char != '\t' && current_char != '\n' && current_char != '\r' && current_char != ':' && current_char != ' ') {error = 1; break;}
            i++;
        }
        else if(state == 1){
            //printf("State 1: %d char: %c\n", i, input_json[end_index-i]);
            if(current_char =='\"' && input_json[end_index-i-1] == '\\'){
                length+=2;
                i+=2;
                continue;
            }
            else if(current_char =='\"'){
                state = 2;
                string_start = i-1;
            }
            length++;
            i++;
        }
        else if(state == 2){
            //printf("There is an error in string at index: %d char: %c\n", i, input_json[end_index-i]);
            if (error != 1){
                //result = string((char*)(input_json-string_start), abs(string_end-string_start)+1);
                //printf("%d, %d, %c, %c\n", string_start, string_end, *(input_json+end_index-string_start), *(input_json+end_index-string_end));
                result.assign((char*)(input_json+end_index-string_start), abs(string_end-string_start)+1);
                //result+='\0';
                //std::cout << result << std::endl;
                return result;
            }
            printf("There is an error in string at index: %d char: %c\n", i, input_json[end_index-i]);
            break;
        }
    }
    if(error == 1 || result.empty()) printf("Couldn't get the string\n");
    return result;
}



string structural_iterator::getValueBackward(int start_index, int end_index, primitive_type& type){
    int i = 1;
    int length = 0;
    int state = 0;
    int error = 0;
    int string_start = 0;
    int string_end = 0;
    string result;
    int temp_end_index = end_index;
    int temp_start_index = start_index;
    char end_char = input_json[temp_end_index-1];
    char start_char = input_json[temp_start_index];
    while(end_char == ' ') {
        temp_end_index--;
        end_char = input_json[temp_end_index];
    }
    while(start_char == ' ') {
        temp_start_index++;
        start_char = input_json[temp_start_index];
    }
    if(end_index <= start_index){
        // error
        printf("no token found!\n");
        return NULL;
    }
    length = temp_end_index - temp_start_index + 1;
    //printf("end_char: %c\n", end_char);
    if(end_char == start_char && end_char == '"'){
        length = temp_end_index - temp_start_index;
        result.assign((char*)(input_json+temp_start_index), abs(length));
        return result;
    
    }
    else if(end_char < 58 && end_char > 47 && start_char < 58 && start_char > 47){
        int tmp_index = temp_start_index;
        while(tmp_index < temp_end_index){
            char current_char = input_json[tmp_index];
            //printf("%c\n", current_char);
            if((current_char <48 || current_char > 57) && current_char != '.'){
                printf("not valid number!\n");
                return NULL;
                //error
            }
            tmp_index++;
        }
        result.assign((char*)(input_json+temp_start_index), abs(length));
        return result;
    }
    else if(end_char == 'e' && (start_char == 't' || start_char == 'f')){
        result.assign((char*)(input_json+temp_start_index), abs(length));
        if(result.compare("true") != 0 && result.compare("false") != 0){
            // error
            printf("not 'true' nor 'false'!\n");
            return NULL;
        }
        return result;
    }
    else if(end_char == 'l' && start_char == 'n'){
        result.assign((char*)(input_json+temp_start_index), abs(length));
        if(result.compare("null") == 0){
            // error
            printf("not 'null'!\n");
            return NULL;
        }
        return result;
    }
    else{
        // error
        printf("invalid token!\n");
        return NULL;
    }
    // while(i>=0){
    //     char current_char = input_json[end_index-i];
    //     if(state == 0){
    //         //printf("State 0: %d char: %c\n", i, input_json[end_index-i]);
    //         string_end = i;
    //         if(current_char == '\t' && current_char == '\n' && current_char == '\r' && current_char == ':' && current_char != ' ') ;
    //         else if(current_char == '"') {string_end++; state = 1; type=STRING;}
    //         else if(current_char > 47 && current_char < 58) {state = 3; type = NUMBER;}
    //         else if(current_char == 'l'){state = 4; type = NULL_TYPE;}
    //         else if(current_char == 'e'){state = 5;}
    //         else{error = 1; state=2;}
    //         i++;
    //     }
    //     else if(state == 1){
    //         //printf("State 1: %d char: %c\n", i, input_json[end_index-i]);
    //         if(current_char =='\"' && input_json[end_index-i-1] == '\\'){
    //             length+=2;
    //             i+=2;
    //             continue;
    //         }
    //         else if(current_char =='"'){
    //             state = 2;
    //             string_start = i-1;
    //         }
    //         length++;
    //         i++;
    //     }
    //     else if(state == 2){
    //         //printf("State 2: %d char: %c\n", i, input_json[end_index-i]);
    //         if (error != 1){
    //             result.assign((char*)(input_json+end_index-string_start), abs(string_end-string_start)+1);

    //             //std::cout << "length is: " << result.length() << ' ' << result << std::endl;
    //             //result+='\0';
    //             return result;
    //         }
    //         printf("There is an error in string at index: %d char: %c\n", i, input_json[end_index-i]);
    //         break;
    //     }
    //     else if(state == 3){
    //         //printf("State 3: %d char: %c\n", i, input_json[end_index-i]);
    //         if(current_char == '\t' && current_char == '\n' && current_char == '\r' || current_char == ':' || current_char == ' ') { 
    //             string_start = i-1;
    //             state = 2;
    //         }
    //         else if((current_char <48 || current_char > 57) && current_char != '.'){
    //             state = 2;
    //             error = 1;
    //         }
    //         length++;
    //         i++;
    //     }
    //     else if(state == 4){
    //         printf("State 4: %d char: %c\n", i, input_json[end_index-i]);
    //         string_start = i+2;
    //         //result = string((char*)(input_json-string_start), abs(string_end-string_start)+1);
    //         result.assign((char*)(input_json+end_index-string_start), abs(string_end-string_start)+1);
    //         //result+='\0';
    //         if(result.compare("null")==0) {return result;}
    //         else error = 1;
    //         break;
    //     }
    //     else if(state == 5){
    //         //printf("State 5: %d char: %c\n", i, input_json[end_index-i]);

    //         string_start = i+3;
    //         int string_start_2 = i+2;

    //         //result = string((char*)(input_json-string_start), abs(string_end-string_start)+1);
    //         result.assign((char*)(input_json+end_index-string_start), abs(string_end-string_start)+1);
    //         //result+='\0';
    //         if(result.compare("false")==0) {return result;}
    //         //result = string((char*)(input_json-string_start_2), abs(string_end-string_start_2)+1);
    //         result.assign((char*)(input_json+end_index-string_start), abs(string_end-string_start)+1);
    //         //result+='\0';
    //         if(result.compare("true")==0) {return result;}
    //         else error = 1;
    //         break;
    //     }
    // }
    // if(error == 1 || result.empty()) printf("Couldn't get the value\n");
    // return result;
}

int structural_iterator::find_specific_key(string input_key){
    int num_child = buf_tree_index[current_array_index];
    int start_index = buf_string_index_length[current_array_index]-1;
    if(input_json[start_index] != '{'){
        printf("error: Not an object %c\n", input_json[start_index]);
        return 0;
    }
    for(int i=1; i<num_child+1; i++){
        int child_index = buf_tree_index[current_array_index+i];
        if(buf_tree_index[child_index] == 0){
            if(i == 1){
                int previous_node_start_index = start_index;
                int child_node_start_index = previous_node_start_index+1;
                int child_node_end_index = buf_string_index_length[child_index]-1;
                int colon_index = buf_string_index_length[current_array_index+i]-1;

                //printf("is comma and first child: %d, %d, %d, ", previous_node_start_index, child_node_start_index, child_node_end_index);
                //printf("%c, %c, %c\n", input_json[previous_node_start_index], input_json[child_node_start_index], input_json[child_node_end_index]);
                string key = getString(child_node_start_index, colon_index);
                //std::cout << key << ' ' << key.compare(input_key) << ' ' << key.length() << ' ' <<  input_key.length() << std::endl;
                if(key.compare(input_key)==0) return i;
                continue;
            }
            else{
                //printf("is comma\n");
                int previous_node_address_index = buf_tree_index[current_array_index+i-1];
                int previous_node_end_index = 0;
                if(buf_tree_index[previous_node_address_index] > 0){
                    int previous_node_start_index = buf_string_index_length[previous_node_address_index]-1;
                    previous_node_end_index = previous_node_start_index + buf_string_index_length[current_array_index+i-1]-1;
                }
                else{
                    previous_node_end_index = buf_string_index_length[previous_node_address_index]-1;
                }
                int child_node_start_index = previous_node_end_index+1;
                int child_node_end_index = buf_string_index_length[child_index]-1;
                int colon_index = buf_string_index_length[current_array_index+i]-1;
                //printf("is comma : %d, %d, ", child_node_start_index, child_node_end_index);
                //printf("%c, %c\n", input_json[child_node_start_index], input_json[child_node_end_index]);
                //printf("%.*s\n", child_node_end_index-child_node_start_index+1, input_json+child_node_start_index);
                string key = getString(child_node_start_index, colon_index);
                //std::cout << key << ' ' << key.compare(input_key) << key.length() << input_key.length() << std::endl;
                if(key.compare(input_key)==0) return i;
                continue;

            }
        }
        else{
            //printf("not a comma\n");
            int child_node_start_index = buf_string_index_length[child_index]-1;
            int child_node_length = buf_string_index_length[current_array_index+i]-1;
            int child_node_end_index = child_node_start_index+child_node_length-1;
            //printf("%.*s\n", child_node_length+1, input_json+child_node_start_index);
            string key = getStringBackward(child_node_start_index);
            //std::cout << key << ' ' << key.compare(input_key) << key.length() << input_key.length() << std::endl;
            if(key.compare(input_key)==0){
                return i;
                break;
            }
            continue;


        }
    }
    printf("child not found\n");
    return 0;

}


string structural_iterator::get_key(){
    int num_child = buf_tree_index[parent_array_index];
    int i =1;
    for(;i<num_child; i++){
        if(buf_tree_index[parent_array_index+i] == current_array_index) break;
    }
    int start_index = buf_string_index_length[parent_array_index]-1;
    if(input_json[start_index] != '{'){
        printf("error: Not an object\n");
        string key;
        return key;
    }
    int child_index = current_array_index;
    if(buf_tree_index[child_index] == 0){
        if(i == 1){
            int previous_node_start_index = start_index;
            int child_node_start_index = previous_node_start_index+1;
            //int child_node_end_index = buf_string_index_length[child_index]-1;
            int colon_index = buf_string_index_length[parent_array_index+i]-1;
            //char colon_char = input_json[colon_index];
            string key = getString(child_node_start_index, colon_index);

            return key;
        }
        else{
            int previous_node_address_index = buf_tree_index[parent_array_index+i-1];
            int previous_node_end_index = 0;
            if(buf_tree_index[previous_node_address_index] > 0){
                int previous_node_start_index = buf_string_index_length[previous_node_address_index]-1;
                previous_node_end_index = previous_node_start_index + buf_string_index_length[parent_array_index+i-1]-1;
            }
            else{
                previous_node_end_index = buf_string_index_length[previous_node_address_index]-1;
            }
            int child_node_start_index = previous_node_end_index+1;
            int colon_index = buf_string_index_length[parent_array_index+i]-1;
            string key = getString(child_node_start_index, colon_index);
            return key;
        }
    }
    else{
        int child_node_start_index = buf_string_index_length[child_index]-1;
        int child_node_length = buf_string_index_length[parent_array_index+i]-1;
        int child_node_end_index = child_node_start_index+child_node_length-1;
        string key = getStringBackward(child_node_start_index);
        return key;

    }
    printf("child not found\n");
    return 0;

}

string structural_iterator::get_value(){
    int num_child = buf_tree_index[parent_array_index];
    int i =1;
    for(;i<num_child; i++){
        if(buf_tree_index[parent_array_index+i] == current_array_index) break;
    }
    int start_index = buf_string_index_length[parent_array_index]-1;
    int child_index = current_array_index;
    if(buf_tree_index[child_index] == 0){
        int child_node_end_index = buf_string_index_length[child_index]-1;
        int child_node_start_index = 0;
        while (input_json[child_node_end_index-1] == '}' || input_json[child_node_end_index-1] == ']')
        {
            child_node_end_index--;
        }
        //printf("last char: %c\n", input_json[child_node_end_index-1]);
        //printf("colon char: %c\n", input_json[buf_string_index_length[parent_array_index+i]-1]);
        int colon_index = buf_string_index_length[parent_array_index+i]-1;
        char colon_char;
        if(colon_index < json_length && (colon_char = input_json[colon_index]) == ':'){
            child_node_start_index = colon_index + 1;
        }
        else{
            if(i == 1){
                int previous_node_start_index = start_index;
                child_node_start_index = previous_node_start_index+1;
            }
            else{
                int previous_node_address_index = buf_tree_index[parent_array_index+i-1];
                int previous_node_end_index = 0;
                if(buf_tree_index[previous_node_address_index] > 0){
                    int previous_node_start_index = buf_string_index_length[previous_node_address_index]-1;
                    previous_node_end_index = previous_node_start_index + buf_string_index_length[parent_array_index+i-1]-1;
                }
                else{
                    previous_node_end_index = buf_string_index_length[previous_node_address_index]-1;
                }
                child_node_start_index = previous_node_end_index+1;
            }
        }
        //printf("the  character is : %c\n", input_json[child_node_end_index]);

        
        primitive_type type;
        string key = getValueBackward(child_node_start_index, child_node_end_index, type);
        return key;
    }
    else{
        token_type type;
        int child_node_start_index = buf_string_index_length[child_index]-1;
        if(input_json[child_node_start_index]== '[') type = ARRAY;
        if(input_json[child_node_start_index]== '{') type = OBJECT;
        int child_node_length = buf_string_index_length[parent_array_index+i];
        int child_node_end_index = child_node_start_index+child_node_length-1;
        string key =  string((char*)input_json+child_node_start_index, child_node_end_index-child_node_start_index+1);
        return key;

    }
    printf("child not found\n");
    return 0;
}

