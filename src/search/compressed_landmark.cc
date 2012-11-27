#include "compressed_landmark.h"

#include "globals.h"
#include "operator.h"

using namespace std;

Landmark::Landmark(vector<RelaxedOperator *> &operators, int cost) :
    vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> >(operators.begin(), operators.end()),
    cost(cost) {
}

inline int get_op_index(const Operator *op) {
    /* TODO: The op_index computation is duplicated from
     LabelReducer::get_op_index() and actually belongs neither
     here nor there. There should be some canonical way of getting
     from an Operator pointer to an index, but it's not clear how to
     do this in a way that best fits the overall planner
     architecture (taking into account axioms etc.) */
    int op_index = op - &*g_operators.begin();
    assert(op_index >= 0 && op_index < g_operators.size());
    return op_index;
}

CompressedLandmark::CompressedLandmark(const std::vector<RelaxedOperator *> &landmark, int cost_)
    : operators(0), length(0), cost(cost_) {
    vector<uint32_t> indices(landmark.size());
    for (size_t i=0; i < landmark.size(); ++i) {
        indices[i] = get_op_index(landmark[i]->op);
    }
    sort(indices.begin(), indices.end());

    // Try if bit array is smaller
    EWAHBoolArray<uint32_t> bit_array;
    for (size_t i=0; i < indices.size(); ++i) {
        bit_array.set(indices[i]);
    }
    const vector<uint32_t> *vector_to_copy;
    if (bit_array.bufferSize() < indices.size()) {
        vector_to_copy = &bit_array.getBuffer();
        is_bit_array = true;
    } else {
        vector_to_copy = &indices;
        is_bit_array = false;
    }
    length = vector_to_copy->size();
    operators = new uint32_t[length];
    std::copy( vector_to_copy->begin(), vector_to_copy->end(), operators);
}
CompressedLandmark::~CompressedLandmark() {
    delete[] operators;
}

Landmark CompressedLandmark::decompress(const std::vector<RelaxedOperator> &relaxed_operators) {
    vector<uint32_t> buffer(operators, operators + length);
    vector<uint32_t> *indices = &buffer;
    vector<uint32_t> set_bits;
    if (is_bit_array) {
        EWAHBoolArray<uint32_t> bit_array;
        bit_array.setBuffer(buffer);
        bit_array.appendSetBits(set_bits);
        indices = &set_bits;
    }
    vector<RelaxedOperator *> ops;
    for (size_t i=0; i < indices->size(); ++i) {
        uint32_t op_i = (*indices)[i];
        RelaxedOperator *op = const_cast<RelaxedOperator *>(&relaxed_operators[op_i]);
        assert(op->op == &g_operators[op_i]);
        ops.push_back(op);
    }
    return Landmark(ops, cost);
}
