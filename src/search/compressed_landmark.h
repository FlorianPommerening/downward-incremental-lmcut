#ifndef COMPRESSED_LANDMARK_H
#define COMPRESSED_LANDMARK_H

#include "lm_cut_heuristic.h"
#include "memory_tracking_allocator.h"
#include <EWAHBoolArray/headers/ewah.h>

#include <boost/shared_ptr.hpp>
#include <vector>

// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark : public vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> > {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

class CompressedLandmark {
    uint32_t* operators;
    bool is_bit_array : 1;
    unsigned short length : 15;
public:
    unsigned short cost;
    CompressedLandmark(const std::vector<RelaxedOperator *> &landmark, int cost_);
    ~CompressedLandmark();
    Landmark decompress(const std::vector<RelaxedOperator> &relaxed_operators);
};

typedef boost::shared_ptr<CompressedLandmark> CompressedLandmarkPtr;

#endif
