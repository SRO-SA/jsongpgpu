
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
using namespace std;

enum tokens_type_enum { 
    OBJECT,
    ARRAY,
    PRIMITIVE
}; 

enum primitive_type_enum { 
    NUMBER,
    TRUE,
    FALSE,
    NULL_TYPE,
    STRING
}; 

typedef tokens_type_enum token_type;
typedef primitive_type_enum primitive_type;


class structural_iterator{
    public:
        const int32_t* buf_tree_index; // array that holds tree
        const int32_t* buf_string_index_length; //array that holds string index
        const uint8_t* input_json;  // JSON string

        int current_array_index = 0;
        int parent_array_index = 0;

        int tree_size;  // total size of tree array
        int json_length;    //size of JSON string

        token_type get_type();
        std::string get_key();
        std::string get_value();
        int find_specific_key(string key);
        int goto_index(int index);
        void reset();
        bool has_key();
        bool has_value();


    structural_iterator(int32_t* buf, uint8_t* json_string,
                      int size, int json_string_length){
        buf_tree_index =buf;
        buf_string_index_length = buf+size;
        tree_size = size;

        input_json = json_string;
        json_length =json_string_length;


    }
      


    private:

        string getString(int start_index, int end_index);
        string getStringBackward(int end_index);
        string getValueBackward(int start_index, int end_index, primitive_type& type);

        int get_value(int index);



        // struct node* get_last_string_index();
        // int32_t* peek_next_node();
        // const int32_t* next_node();
        // int32_t* current_node();
        // void update();
};
