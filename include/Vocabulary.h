//
// Created by hanfuyong on 2022/8/22.
//

#ifndef NAIVE_SLAM_VOCABULARY_H
#define NAIVE_SLAM_VOCABULARY_H

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

namespace Naive_SLAM{

    typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> Vocabulary;

}

#endif //NAIVE_SLAM_VOCABULARY_H
