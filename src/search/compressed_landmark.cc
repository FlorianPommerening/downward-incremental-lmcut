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

CompressedLandmark::CompressedLandmark(const std::vector<RelaxedOperator *> &landmark, uint32_t cost)
    : data(0) {
    vector<uint32_t> indices(landmark.size());
    for (size_t i=0; i < landmark.size(); ++i) {
        indices[i] = get_op_index(landmark[i]->op);
    }
    sort(indices.begin(), indices.end());

    Header header;
    header.cost = cost;
    header.reference_count = 1;

    // Try if bit array is smaller
    EWAHBoolArray<uint32_t> bit_array;
    for (size_t i=0; i < indices.size(); ++i) {
        bit_array.set(indices[i]);
    }
    const vector<uint32_t> *vector_to_copy;
    if (bit_array.bufferSize() < indices.size()) {
        vector_to_copy = &bit_array.getBuffer();
        header.is_bit_array = true;
    } else {
        vector_to_copy = &indices;
        header.is_bit_array = false;
    }

    header.length = vector_to_copy->size();
    size_t header_length = sizeof(header) / sizeof(uint32_t);
    assert(header_length == 3);
    data = new uint32_t[header_length + header.length];
    *((Header *) data) = header;
    std::copy(vector_to_copy->begin(), vector_to_copy->end(), data + header_length);
}

CompressedLandmark::CompressedLandmark(const CompressedLandmark& other)
    : data(other.data) {
    data[0]++;
}

CompressedLandmark::~CompressedLandmark() {
    data[0]--;
    if (!data[0]) {
        delete[] data;
    }
}

Landmark CompressedLandmark::decompress(const std::vector<RelaxedOperator> &relaxed_operators) {
    Header header = *((Header*) data);
    size_t header_length = sizeof(header) / sizeof(uint32_t);
    assert(header_length == 3);
    vector<uint32_t> buffer(data + header_length, data + header_length + header.length);
    vector<uint32_t> *indices = &buffer;
    vector<uint32_t> set_bits;
    if (header.is_bit_array) {
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
    return Landmark(ops, header.cost);
}

uint32_t CompressedLandmark::get_cost() {
    return data[1];
}
