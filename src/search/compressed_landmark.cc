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
        assert(indices[i] < RELAXED_OPERATORS->size());
    }
    sort(indices.begin(), indices.end());

    // Try if bit array is smaller than storing indices
    Header header;
    header.cost = cost;
    header.reference_count = 1;
    // Calculate size of index array without computing it and use it as abound for bit array.
    int compressed_indeces_size = (landmark.size() * OPERATOR_WORD_LENGTH + 31) / 32;
    bool bit_array_better = compress_bit_array(header, indices, compressed_indeces_size);
    if (!bit_array_better) {
        compress_index_set(header, indices, compressed_indeces_size);
    }
}

CompressedLandmark::CompressedLandmark(const CompressedLandmark& other)
    : data(other.data) {
    data[0]++;
}

CompressedLandmark::~CompressedLandmark() {
    data[0]--;
    if (!data[0]) {
        Header header = *((Header*) data);
        if (header.is_bit_array) {
            g_memory_tracking_allocated -= (3 + header.length) * sizeof(uint32_t);
            g_memory_tracking_landmarks_bit_array -= (3 + header.length) * sizeof(uint32_t);
        } else {
            int compressed_indeces_size = (header.length * OPERATOR_WORD_LENGTH + 31) / 32;
            g_memory_tracking_allocated -= (3 + compressed_indeces_size) * sizeof(uint32_t);
            g_memory_tracking_landmarks_indices -= (3 + compressed_indeces_size) * sizeof(uint32_t);
        }
        delete[] data;
    }
}

Landmark CompressedLandmark::decompress() {
    Header header = *((Header*) data);
    size_t header_length = sizeof(header) / sizeof(uint32_t);
    assert(header_length == 3);
    vector<uint32_t> indices;
    if (header.is_bit_array) {
        decompress_bit_array(header, data + header_length, indices);
    } else {
        decompress_index_set(header, data + header_length, indices);
    }
    vector<RelaxedOperator *> ops;
    for (size_t i=0; i < indices.size(); ++i) {
        const RelaxedOperator &const_op = (*RELAXED_OPERATORS)[indices[i]];
        RelaxedOperator *op = const_cast<RelaxedOperator *>(&const_op);
        assert(op->op == &g_operators[indices[i]]);
        ops.push_back(op);
    }
    return Landmark(ops, header.cost);
}

void CompressedLandmark::set_operators(const std::vector<RelaxedOperator> &relaxed_operators_) {
    RELAXED_OPERATORS = &relaxed_operators_;
    uint32_t size = RELAXED_OPERATORS->size();
    OPERATOR_WORD_LENGTH = 0;
    MASK = 0;
    while (size) {
        ++OPERATOR_WORD_LENGTH;
        MASK <<= 1;
        MASK |= 1;
        size >>= 1;
    }
}

bool CompressedLandmark::compress_index_set(Header &header, const std::vector<uint32_t> &indices, int max_size) {
    int compressed_indeces_size = (indices.size() * OPERATOR_WORD_LENGTH + 31) / 32;
    if (compressed_indeces_size > max_size) {
        return false;
    }
    header.is_bit_array = false;
    // store number of operators (length of data can be calculated from that)
    header.length = indices.size();
    assert(sizeof(header) / sizeof(uint32_t) == 3);

    data = new uint32_t[3 + compressed_indeces_size];
    g_memory_tracking_allocated += (3 + compressed_indeces_size) * sizeof(uint32_t);
    g_memory_tracking_landmarks_indices += (3 + compressed_indeces_size) * sizeof(uint32_t);

    *((Header *) data) = header;

    unsigned char *p_current_8 = (unsigned char *) (data + 3);
    uint64_t *p_current_64 = (uint64_t *) p_current_8;
    memset(p_current_8, 0, sizeof(uint32_t) * compressed_indeces_size);
    int offset = 0;
    for (size_t i=0; i < indices.size(); ++i) {
        while (offset >= 8) {
            ++p_current_8;
            assert((uint32_t *) p_current_8 < (data + 3 + compressed_indeces_size));
            p_current_64 = (uint64_t *) p_current_8;
            offset -= 8;
        }
        offset += OPERATOR_WORD_LENGTH;
        uint64_t index = indices[i];
        uint64_t big_endian_index = __builtin_bswap64(index << (64 - offset));
        *p_current_64 |= big_endian_index;
    }
    return true;
}

void CompressedLandmark::decompress_index_set(const Header& header, uint32_t *p_buffer, vector<uint32_t> &indices) {
    indices.resize(header.length);
    unsigned char *p_current_8 = (unsigned char *) p_buffer;
    uint64_t *p_current_64 = (uint64_t *) p_current_8;
    int offset = 0;
    for (size_t i=0; i < header.length; ++i) {
        while (offset >= 8) {
            ++p_current_8;
            assert((uint32_t *) p_current_8 < (data + 3 + ((header.length * OPERATOR_WORD_LENGTH + 31) / 32)));
            p_current_64 = (uint64_t *) p_current_8;
            offset -= 8;
        }
        uint64_t big_endian_current = __builtin_bswap64(*p_current_64);
        uint64_t big_endian_id = big_endian_current >> (64 - (OPERATOR_WORD_LENGTH + offset));
        big_endian_id &= MASK;
        indices[i] = big_endian_id;
        offset += OPERATOR_WORD_LENGTH;
    }
}

bool CompressedLandmark::compress_bit_array(Header &header, const std::vector<uint32_t> &indices, int max_size) {
    EWAHBoolArray<uint32_t> bit_array;
    for (size_t i=0; i < indices.size(); ++i) {
        bit_array.set(indices[i]);
    }
    const vector<uint32_t> &buffer = bit_array.getBuffer();
    if (buffer.size() > max_size) {
        return false;
    }
    header.is_bit_array = true;
    header.length = buffer.size();
    assert(sizeof(header) / sizeof(uint32_t) == 3);

    data = new uint32_t[3 + header.length];
    g_memory_tracking_allocated += (3 + header.length) * sizeof(uint32_t);
    g_memory_tracking_landmarks_bit_array += (3 + header.length) * sizeof(uint32_t);
    *((Header *) data) = header;
    std::copy(buffer.begin(), buffer.end(), data + 3);
    return true;
}

void CompressedLandmark::decompress_bit_array(const Header& header, uint32_t *p_buffer, vector<uint32_t> &indices) {
    vector<uint32_t> buffer(p_buffer, p_buffer + header.length);
    EWAHBoolArray<uint32_t> bit_array;
    bit_array.setBuffer(buffer);
    bit_array.appendSetBits(indices);
}

const std::vector<RelaxedOperator> *CompressedLandmark::RELAXED_OPERATORS = 0;
uint32_t CompressedLandmark::MASK = 0;
int CompressedLandmark::OPERATOR_WORD_LENGTH = 0;
CompressedLandmark CompressedLandmark::EMPTY = CompressedLandmark(vector<RelaxedOperator*>(), -1);
