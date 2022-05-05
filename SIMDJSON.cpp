#include "simdjson.h"
#include <stdio.h>
#include <time.h>

using namespace simdjson;
    double calcTime();

int main(){
    double meantime = 0;
    for (int i=0;i<5;i++){
        meantime = meantime + calcTime();
    }
    std::cout << meantime/5 << std::endl;
    return 0;
}

double calcTime(){
    simdjson::dom::parser parser(4000000000);
    //parser.set_max_capacity(4000000000);
    dom::document_stream docs;
    time_t start, end;
    time_t pstart, pend;
    simdjson::error_code error;
    double inttotal = 0;
    int count;
    start = clock();
    parser.load_many("./../Large-Json/google_map_small_records.json").get(docs);
    //parser.load_many("./inputs/Tokenizer_long_6000.txt").get(docs);
    //printf("max_capaacity: %ld\n", parser.max_capacity());
    count = 0;
    pstart = clock();
    for (auto doc : docs) {
        error = doc.error();
        if (error) {
            std::wcerr << "Parsing failed with: " << error << std::endl;
            exit(1);
        }
        count++;
        /*for(auto iter = doc.begin(); iter!=doc.end(); iter++){
            //bool completed = (*iter).at(0);
            pstart = clock();
            std::cout << (*iter) << std::endl;
            pend = clock();
            inttotal += ((double)(pend-pstart)/CLOCKS_PER_SEC)*1000;

        }*/
    //std::cout << "HERE" << std::endl;
    }
    pend = clock();
    end = clock();
    std::cout << "loop: " << ((double)(pend-pstart)/CLOCKS_PER_SEC)*1000 << std::endl;
    std::cout << "total: " << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    return ((double)(end-start)/CLOCKS_PER_SEC)*1000;// - inttotal;
}

