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
    // current_array_index = 0;
    // parent_array_index = 0;
    node = 0;
    node_depth = 1;
    node_type = OBJECT;
}

int structural_iterator::goto_array_index(int i){
    int total= i;
    int next_node = node+1;
    node_depth = buf_json_depth[node];
    int next_node_depth = buf_json_depth[next_node];
    while(total != 1 && !(next_node_depth < node_depth)){
        int string_index = buf_json_start_in_string[next_node]-1;
        char node_char = input_json[string_index];
        // printf("next node: %d,  %d previous char %c\n", next_node, node_depth, input_json[buf_json_start_in_string[next_node-1]-1]);
        // printf("current char %c\n", input_json[buf_json_start_in_string[next_node]-1]);

        if(node_char == '[' || node_char == '{'){
            int jump = buf_json_other_index[next_node];
            next_node = jump;
        }
        if(node_char == ',' || node_char == '\n'){
            // printf("%d current char %c\n", node_depth, node_char);
            total--;
        }
        next_node++;
        next_node_depth = buf_json_depth[next_node];
    }
    //printf("%d current char %c\n", next_node_depth, input_json[buf_json_start_in_string[next_node]-1]);
    // printf("--- %d current char %c\n", buf_json_depth[next_node-1], input_json[buf_json_start_in_string[next_node-1]-1]);
    // printf("=== %d current char %c\n", buf_json_depth[next_node+1], input_json[buf_json_start_in_string[next_node+1]-1]);

    if(total == 1){
        node = next_node;
        node_depth = next_node_depth;
        // printf("NODE DEPTH: %d\n", node_depth);
        int string_index = buf_json_start_in_string[node]-1;
        char node_char = input_json[string_index];
        if(node_char == '{'){
            node_type = OBJECT;
        }
        else if(node_char == '['){
            node_type = ARRAY;
        }
        else if(node_char == ',' || node_char == '\n'){
            node_type = VALUE;
        }
        return node;
    }
    return 0;
}

int structural_iterator::goto_index(int i){
    // int index = i;
    // if(buf_tree_index[current_array_index] < index){
    //     printf("error on setting new index\n");
    //     return 0;
    // }
    // parent_array_index = current_array_index;
    // int child_index = buf_tree_index[current_array_index+index];
    // current_array_index = child_index;
    // //printf("parent is: %d and current node is %d\n", parent_array_index, current_array_index);
    // return current_array_index;
    //printf("%d current char %c\n", node, input_json[buf_json_start_in_string[node]-1]);
    node = node+i;
    node_depth = buf_json_depth[node];
    int string_index = buf_json_start_in_string[node]-1;
    char node_char = input_json[string_index];
    if(node_char == ':'){
        node++;
        node_depth = buf_json_depth[node];
        string_index = buf_json_start_in_string[node]-1;
        node_char = input_json[string_index];
    }
    if(node_char == ']' || node_char == '}'){
        node--;
        node_depth = buf_json_depth[node];
        string_index = buf_json_start_in_string[node]-1;
        node_char = input_json[string_index];
    }

    //printf("%d current char %c\n", node, node_char);
    if(node_char == '{'){
        node_type = OBJECT;
    }
    else if(node_char == '['){
        node_type = ARRAY;
    }
    else if(node_char == ',' || node_char == '\n'){
        node_type = VALUE;
    }
    else if(node_char == ':'){
        node_type = KEYVALUE;
    }
    else if(node_char == ']' || node_char == '}'){
        node++;
        node_depth = buf_json_depth[node];
        node_type = CLOSING;
    }
    else{
        return 0;
    }
    return i;
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
    char end_char = input_json[temp_end_index];
    char start_char = input_json[temp_start_index];
    while(end_char == ' ' || end_char == '}' || end_char == ']') {
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
    // printf("start_char: %c end_char: %c\n", start_char, end_char);
    if(end_char == start_char && end_char == '"'){
        length = temp_end_index - temp_start_index - 1;
        result.assign((char*)(input_json+temp_start_index+1), abs(length));
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
        std::cout << result << endl;
        if(result.compare("null") != 0){
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
    int node_string_index = buf_json_start_in_string[node]-1;
    char node_char = input_json[node_string_index];
    // printf("%c\n", node_char);
    if(node_char == ':'){
        // printf("is colon\n");
        goto_index(1);
    }
    if(node_type != OBJECT){
        cout << "Node is not an object" << endl;
        return 0;
    }
    
    int end_index = buf_json_other_index[node];
    // printf("begin: %d, end: %d, node depth: %d real depth: %d\n", node, end_index, node_depth, buf_json_depth[node]);
    int i = node+1;
    while (i < end_index)
    {
        int depth = buf_json_depth[i];
        int current_string_index = buf_json_start_in_string[i]-1;
        // printf("depth: %d, char: %c\n", depth, input_json[current_string_index]);
        if(node_depth < depth){
            int end = buf_json_other_index[i];
            current_string_index = buf_json_start_in_string[i]-1;
            // printf("end index: %d, depth: %d, char: %c\n", end, depth, input_json[current_string_index]);
            i = end;
        }
        if(node_depth > depth) break;
        int string_index = buf_json_start_in_string[i]-1;
        char current_primitive_char = input_json[string_index];
        if(current_primitive_char == ':'){
            string key;
            int previous_index = i-1;
            int previous_char_index = buf_json_start_in_string[previous_index]-1;
            key = getString(previous_char_index+1, string_index);
            //cout << key <<  " " << i << endl;
            if(key.compare(input_key)==0){return i-node;}
            i++;
            continue;
        }
        i++;
    }
    return 0;
    
    // int num_child = buf_tree_index[current_array_index];
    // int start_index = buf_string_index_length[current_array_index]-1;
    // if(input_json[start_index] != '{'){
    //     printf("error: Not an object %c\n", input_json[start_index]);
    //     return 0;
    // }
    // for(int i=1; i<num_child+1; i++){
    //     int child_index = buf_tree_index[current_array_index+i];
    //     if(buf_tree_index[child_index] == 0){
    //         if(i == 1){
    //             int previous_node_start_index = start_index;
    //             int child_node_start_index = previous_node_start_index+1;
    //             int child_node_end_index = buf_string_index_length[child_index]-1;
    //             int colon_index = buf_string_index_length[current_array_index+i]-1;

    //             //printf("is comma and first child: %d, %d, %d, ", previous_node_start_index, child_node_start_index, child_node_end_index);
    //             //printf("%c, %c, %c\n", input_json[previous_node_start_index], input_json[child_node_start_index], input_json[child_node_end_index]);
    //             string key = getString(child_node_start_index, colon_index);
    //             //std::cout << key << ' ' << key.compare(input_key) << ' ' << key.length() << ' ' <<  input_key.length() << std::endl;
    //             if(key.compare(input_key)==0) return i;
    //             continue;
    //         }
    //         else{
    //             //printf("is comma\n");
    //             int previous_node_address_index = buf_tree_index[current_array_index+i-1];
    //             int previous_node_end_index = 0;
    //             if(buf_tree_index[previous_node_address_index] > 0){
    //                 int previous_node_start_index = buf_string_index_length[previous_node_address_index]-1;
    //                 previous_node_end_index = previous_node_start_index + buf_string_index_length[current_array_index+i-1]-1;
    //             }
    //             else{
    //                 previous_node_end_index = buf_string_index_length[previous_node_address_index]-1;
    //             }
    //             int child_node_start_index = previous_node_end_index+1;
    //             int child_node_end_index = buf_string_index_length[child_index]-1;
    //             int colon_index = buf_string_index_length[current_array_index+i]-1;
    //             //printf("is comma : %d, %d, ", child_node_start_index, child_node_end_index);
    //             //printf("%c, %c\n", input_json[child_node_start_index], input_json[child_node_end_index]);
    //             //printf("%.*s\n", child_node_end_index-child_node_start_index+1, input_json+child_node_start_index);
    //             string key = getString(child_node_start_index, colon_index);
    //             //std::cout << key << ' ' << key.compare(input_key) << key.length() << input_key.length() << std::endl;
    //             if(key.compare(input_key)==0) return i;
    //             continue;

    //         }
    //     }
    //     else{
    //         //printf("not a comma\n");
    //         int child_node_start_index = buf_string_index_length[child_index]-1;
    //         int child_node_length = buf_string_index_length[current_array_index+i]-1;
    //         int child_node_end_index = child_node_start_index+child_node_length-1;
    //         //printf("%.*s\n", child_node_length+1, input_json+child_node_start_index);
    //         string key = getStringBackward(child_node_start_index);
    //         //std::cout << key << ' ' << key.compare(input_key) << key.length() << input_key.length() << std::endl;
    //         if(key.compare(input_key)==0){
    //             return i;
    //             break;
    //         }
    //         continue;


    //     }
    // }
    // printf("child not found\n");
    // return 0;

}


string structural_iterator::get_key(){

    int char_index = buf_json_start_in_string[node]-1;
    char node_char = input_json[char_index];
    string key;

    if(node_char == '[' || node_char == '{' || node_char == ',' || node_char == '\n'){
        int previous_index = node - 1;
        int key_char_index = buf_json_start_in_string[previous_index]-1;
        char key_char = input_json[key_char_index];
        if(key_char == ':'){
            int previous_previous_index = previous_index - 1;
            int key_start_char_index = buf_json_start_in_string[previous_previous_index]-1;
            key = getString(key_start_char_index+1, key_char_index);
            return key;
        }
        else {
            return key;
        }
        if(node_char == ':'){
            int previous_index = node - 1;
            int key_char_index = buf_json_start_in_string[previous_index]-1;
            key = getString(key_char_index+1, char_index);
            return key;
        }
        return key;
    }
    return key;
    // int num_child = buf_tree_index[parent_array_index];
    // int i =1;
    // for(;i<num_child; i++){
    //     if(buf_tree_index[parent_array_index+i] == current_array_index) break;
    // }
    // int start_index = buf_string_index_length[parent_array_index]-1;
    // if(input_json[start_index] != '{'){
    //     printf("error: Not an object\n");
    //     string key;
    //     return key;
    // }
    // int child_index = current_array_index;
    // if(buf_tree_index[child_index] == 0){
    //     if(i == 1){
    //         int previous_node_start_index = start_index;
    //         int child_node_start_index = previous_node_start_index+1;
    //         //int child_node_end_index = buf_string_index_length[child_index]-1;
    //         int colon_index = buf_string_index_length[parent_array_index+i]-1;
    //         //char colon_char = input_json[colon_index];
    //         string key = getString(child_node_start_index, colon_index);

    //         return key;
    //     }
    //     else{
    //         int previous_node_address_index = buf_tree_index[parent_array_index+i-1];
    //         int previous_node_end_index = 0;
    //         if(buf_tree_index[previous_node_address_index] > 0){
    //             int previous_node_start_index = buf_string_index_length[previous_node_address_index]-1;
    //             previous_node_end_index = previous_node_start_index + buf_string_index_length[parent_array_index+i-1]-1;
    //         }
    //         else{
    //             previous_node_end_index = buf_string_index_length[previous_node_address_index]-1;
    //         }
    //         int child_node_start_index = previous_node_end_index+1;
    //         int colon_index = buf_string_index_length[parent_array_index+i]-1;
    //         string key = getString(child_node_start_index, colon_index);
    //         return key;
    //     }
    // }
    // else{
    //     int child_node_start_index = buf_string_index_length[child_index]-1;
    //     int child_node_length = buf_string_index_length[parent_array_index+i]-1;
    //     int child_node_end_index = child_node_start_index+child_node_length-1;
    //     string key = getStringBackward(child_node_start_index);
    //     return key;

    // }
    // printf("child not found\n");
    // return 0;

}

string structural_iterator::get_value(){
    int char_index = buf_json_start_in_string[node]-1;
    char node_char = input_json[char_index];
    string value;
    primitive_type type;
    // printf("value char %c\n", node_char);
    if(node_char == '[' || node_char == '{'){
        int end = buf_json_other_index[node];
        int end_char_index = buf_json_start_in_string[end]-1;
        value = string((char*)input_json+char_index, end_char_index-char_index+1);
        return value;
    }
    else if(node_char == ':'){
        int end = node + 1;
        int end_char_index = buf_json_start_in_string[end]-1;
        // printf("start: %d, end: %d\n", char_index, end_char_index);
        if(input_json[end_char_index] == ',' || input_json[end_char_index] == ']' || input_json[end_char_index] == '}' || input_json[end_char_index] == '\n'){
            // printf("start: %c, end: %c\n", input_json[char_index+1], input_json[end_char_index-1]);
            value = getValueBackward(char_index+1, end_char_index, type);
        }
        else if(input_json[end_char_index] == '['){
            int start = end;
            int start_char_index = buf_json_start_in_string[start] - 1;
            end = node + 2;
            end_char_index = buf_json_start_in_string[end] - 1;
            // printf("start: %c, end: %c\n", input_json[start_char_index], input_json[end_char_index]);
            value = getValueBackward(start_char_index+1, end_char_index-1, type);
        }
        return value;
    }
    else if(node_char == ',' || node_char == '\n'){
        int start = node - 1;
        int start_char_index = buf_json_start_in_string[start]-1;
        value = getValueBackward(start_char_index+1, char_index-1, type);
        return value;
    }
    return value;
    // int num_child = buf_tree_index[parent_array_index];
    // int i =1;
    // for(;i<num_child; i++){
    //     if(buf_tree_index[parent_array_index+i] == current_array_index) break;
    // }
    // int start_index = buf_string_index_length[parent_array_index]-1;
    // int child_index = current_array_index;
    // if(buf_tree_index[child_index] == 0){
    //     int child_node_end_index = buf_string_index_length[child_index]-1;
    //     int child_node_start_index = 0;
    //     while (input_json[child_node_end_index-1] == '}' || input_json[child_node_end_index-1] == ']')
    //     {
    //         child_node_end_index--;
    //     }
    //     //printf("last char: %c\n", input_json[child_node_end_index-1]);
    //     //printf("colon char: %c\n", input_json[buf_string_index_length[parent_array_index+i]-1]);
    //     int colon_index = buf_string_index_length[parent_array_index+i]-1;
    //     char colon_char;
    //     if(colon_index < json_length && (colon_char = input_json[colon_index]) == ':'){
    //         child_node_start_index = colon_index + 1;
    //     }
    //     else{
    //         if(i == 1){
    //             int previous_node_start_index = start_index;
    //             child_node_start_index = previous_node_start_index+1;
    //         }
    //         else{
    //             int previous_node_address_index = buf_tree_index[parent_array_index+i-1];
    //             int previous_node_end_index = 0;
    //             if(buf_tree_index[previous_node_address_index] > 0){
    //                 int previous_node_start_index = buf_string_index_length[previous_node_address_index]-1;
    //                 previous_node_end_index = previous_node_start_index + buf_string_index_length[parent_array_index+i-1]-1;
    //             }
    //             else{
    //                 previous_node_end_index = buf_string_index_length[previous_node_address_index]-1;
    //             }
    //             child_node_start_index = previous_node_end_index+1;
    //         }
    //     }
    //     //printf("the  character is : %c\n", input_json[child_node_end_index]);

        
    //     primitive_type type;
    //     string key = getValueBackward(child_node_start_index, child_node_end_index, type);
    //     return key;
    // }
    // else{
    //     token_type type;
    //     int child_node_start_index = buf_string_index_length[child_index]-1;
    //     if(input_json[child_node_start_index]== '[') type = ARRAY;
    //     if(input_json[child_node_start_index]== '{') type = OBJECT;
    //     int child_node_length = buf_string_index_length[parent_array_index+i];
    //     int child_node_end_index = child_node_start_index+child_node_length-1;
    //     string key =  string((char*)input_json+child_node_start_index, child_node_end_index-child_node_start_index+1);
    //     return key;

    // }
    // printf("child not found\n");
    // return 0;
}

