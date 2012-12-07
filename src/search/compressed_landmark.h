#ifndef COMPRESSED_LANDMARK_H
#define COMPRESSED_LANDMARK_H

#include "lm_cut_heuristic.h"
#include "memory_tracking_allocator.h"

#include <EWAHBoolArray/headers/ewah.h>
#include <vector>

// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark : public vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> > {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

// Stores one landmark in a continous area of memory.
// Format:
// | 32 bit: reference counter                    |
// | 32 bit: cost of landmark                     |
// | 1 bit: is_bit_array | 31 bit: length of data |
// | variable length data                         |
class CompressedLandmark {
    struct Header {
        uint32_t reference_count;
        uint32_t cost;
        bool is_bit_array: 1;
        uint32_t length: 31;
    };

    uint32_t* data;
    bool compress_index_set(Header &header, const std::vector<uint32_t> &indices, int max_size);
    void decompress_index_set(const Header& header, uint32_t *p_buffer, vector<uint32_t> &indices);
    bool compress_bit_array(Header &header, const std::vector<uint32_t> &indices, int max_size);
    void decompress_bit_array(const Header& header, uint32_t *p_buffer, vector<uint32_t> &indices);
    static const std::vector<RelaxedOperator> *RELAXED_OPERATORS;
    static uint32_t MASK;
    static int OPERATOR_WORD_LENGTH;
public:
    uint32_t get_cost() {
        return data[1];
    }
    CompressedLandmark(const std::vector<RelaxedOperator *> &landmark, uint32_t cost_);
    CompressedLandmark(const CompressedLandmark& other);
    ~CompressedLandmark();
    Landmark decompress();
    static void set_operators(const std::vector<RelaxedOperator> &relaxed_operators_);
    static CompressedLandmark EMPTY;
};

#endif
