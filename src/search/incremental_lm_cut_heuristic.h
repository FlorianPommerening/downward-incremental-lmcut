#ifndef INCREMENTAL_LM_CUT_HEURISTIC_H
#define INCREMENTAL_LM_CUT_HEURISTIC_H

#include "lm_cut_heuristic.h"
#include "state_proxy.h"
#include "search_space.h"
#include "compressed_landmark.h"
#include <ext/hash_map>


typedef vector<CompressedLandmark, MemoryTrackingAllocator<CompressedLandmark> > SavedLandmarks;



typedef __gnu_cxx::hash_map<StateProxy,
                            SavedLandmarks,
                            __gnu_cxx::hash<StateProxy>,
                            equal_to<StateProxy>,
                            MemoryTrackingAllocator<std::pair<const StateProxy,
                                                              SavedLandmarks> > >
        SavedLandmarksMap;

template<class T1, class T2>
struct ComparePairByFirstComponent {
    bool operator()(pair<T1, T2> const &lhs, pair<T1, T2> const &rhs) {
        return lhs.first < rhs.first;
    }
};

class IncrementalLandmarkCutHeuristic : public LandmarkCutHeuristic {
private:
    // special case for local computation: avoid compressing and decompressing landmarks
    bool local_computation_on;
    bool reevaluating_parent;

    bool keep_frontier;
    bool reevaluate_parent;
    unsigned long memory_limit;
    SavedLandmarksMap saved_landmarks_map;
    void remove_meta_information(StateProxy &state_proxy) {
        saved_landmarks_map.erase(state_proxy);
    }
    std::vector<Landmark> current_decompressed_landmarks;
    std::vector<std::pair<Landmark, CompressedLandmark> > parent_landmarks;
    StateProxy current_state_proxy;
    int current_landmarks_cost;
    StateProxy current_parent_state_proxy;
    bool get_parent_landmarks(const State &parent_state);
protected:
    virtual void initialize();
    virtual bool reach_state(const State &parent_state, const Operator &op,
                             const State &state);
    virtual void finished_state(const State &state, int f, bool is_on_frontier);

    virtual void discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost);
    virtual void reset_operator_costs(const State &state);
    virtual int compute_heuristic(const State &state);
public:
    IncrementalLandmarkCutHeuristic(const Options &opts);
    virtual ~IncrementalLandmarkCutHeuristic();
    virtual void free_up_memory(SearchSpace &search_space);
};


#endif
