//
// Created by hanfuyong on 2022/8/25.
//

#include "Vocabulary.h"

#include <time.h>

using namespace std;

bool load_as_text(Naive_SLAM::Vocabulary* voc, const std::string infile) {
    clock_t tStart = clock();
    bool res = voc->loadFromTextFile(infile);
    printf("Loading fom text: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return res;
}

void load_as_xml(Naive_SLAM::Vocabulary* voc, const std::string infile) {
    clock_t tStart = clock();
    voc->load(infile);
    printf("Loading fom xml: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void load_as_binary(Naive_SLAM::Vocabulary* voc, const std::string infile) {
    clock_t tStart = clock();
    voc->loadFromBinaryFile(infile);
    printf("Loading fom binary: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void save_as_xml(Naive_SLAM::Vocabulary* voc, const std::string outfile) {
    clock_t tStart = clock();
    voc->save(outfile);
    printf("Saving as xml: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void save_as_text(Naive_SLAM::Vocabulary* voc, const std::string outfile) {
    clock_t tStart = clock();
    voc->saveToTextFile(outfile);
    printf("Saving as text: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void save_as_binary(Naive_SLAM::Vocabulary* voc, const std::string outfile) {
    clock_t tStart = clock();
    voc->saveToBinaryFile(outfile);
    printf("Saving as binary: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}


int main(int argc, char **argv) {
    cout << "BoW load/save benchmark" << endl;
    Naive_SLAM::Vocabulary* voc = new Naive_SLAM::Vocabulary();

    load_as_text(voc, std::string(argv[1]));
    save_as_binary(voc, std::string(argv[2]));

    return 0;
}