#include "simdjson.h"
#include <stdio.h>
#include <time.h>
#include <string>

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
    clock_t query_start, query_end;
    simdjson::error_code error;
    double inttotal = 0;
    double query_runtime = 0;
    int count;
    start = clock();
    parser.load_many("./../Large-Json/google_map_small_records.json").get(docs);
    //parser.load_many("./inputs/Tokenizer_long_6000.txt").get(docs);
    //printf("max_capaacity: %ld\n", parser.max_capacity());
    count = 0;
    int which = 0;
    pstart = clock();
    for (auto doc : docs) {
        error = doc.error();
        if (error) {
            std::wcerr << "Parsing failed with: " << error << std::endl;
            exit(1);
        }
        count++;
        query_start = clock();
        simdjson::simdjson_result<simdjson::dom::element> query_res;
        simdjson::simdjson_result<simdjson::dom::element> query_res1;

        switch(which){
            case 0:
            query_res = doc["routes"].at(0)["overview_polyline"]["points"];
            break;
            case 1:
            query_res = doc["aliases"]["zh-hant"].at(1)["value"];
            break;
            case 2:
            query_res = doc.at(1);
            break;
            case 3:
            query_res = doc["user"]["location"];
            break;
            case 4:
            query_res1 = doc["salePrice"];
            break;
        }
        query_end = clock();
        query_runtime = ((double)(query_end-query_start)/CLOCKS_PER_SEC)*1000;
        auto error = query_res.is_string();
        auto error2 = query_res1.is_double();
        //std::cout << "HERE!!!!! " << error << std::endl;
        if(error == 0 && which < 4) continue;
        if(error2 == 0 && which == 4) continue;
        //if(which == 4) std::cout << query_res1.value() << " \nruntime: " << query_runtime << std::endl;
        //if(which < 4)std::cout << query_res.value() << " \nruntime: " << query_runtime << std::endl;

        /*for(auto iter = doc.begin(); iter!=doc.end(); iter++){
            //bool completed = (*iter).at(0);
            pstart = clock();
            std::cout << (*iter) << std::endl;
            pend = clock();
            int total += ((double)(pend-pstart)/CLOCKS_PER_SEC)*1000;

        }*/
    //std::cout << "HERE" << std::endl;
    }
    pend = clock();
    end = clock();
    std::cout << "loop: " << ((double)(pend-pstart)/CLOCKS_PER_SEC)*1000 << std::endl;
    std::cout << "total: " << ((double)(end-start)/CLOCKS_PER_SEC)*1000 << std::endl;

    return ((double)(end-start)/CLOCKS_PER_SEC)*1000;// - inttotal;
}

