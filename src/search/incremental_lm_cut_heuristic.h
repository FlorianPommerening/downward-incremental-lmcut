#ifndef INCREMENTAL_LM_CUT_HEURISTIC_H
#define INCREMENTAL_LM_CUT_HEURISTIC_H

#include "lm_cut_heuristic.h"
#include "state_handle.h"
#include "search_space.h"
#include "utilities.h"
#include "memory_tracking_allocator.h"

#include <boost/shared_ptr.hpp>
#include <ext/hash_map>

// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark : public std::vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> > {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

typedef boost::shared_ptr<Landmark> LandmarkPtr;
typedef std::vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> > SavedLandmarks;



typedef __gnu_cxx::hash_map<StateHandle,
                            SavedLandmarks,
                            __gnu_cxx::hash<StateHandle>,
                            std::equal_to<StateHandle>,
                            MemoryTrackingAllocator<std::pair<const StateHandle,
                                                              SavedLandmarks> > >
        SavedLandmarksMap;

template<class T1, class T2>
struct ComparePairByFirstComponent {
    bool operator()(std::pair<T1, T2> const &lhs, std::pair<T1, T2> const &rhs) {
        return lhs.first < rhs.first;
    }
};

class IncrementalLandmarkCutHeuristic : public LandmarkCutHeuristic, public MemoryReleaser {
private:
    bool local_incremental_computation;
    bool landmarks_belong_to_parent;
    bool keep_frontier;
    bool reevaluate_parent;
    unsigned long memory_limit;
    StateHandle current_state_handle;
    int current_landmarks_cost;
    SavedLandmarks current_landmarks;
    StateHandle current_parent_state_handle;
    SavedLandmarks current_parent_landmarks;
    SavedLandmarksMap saved_landmarks_map;
    void remove_meta_information(StateHandle &state_handle) {
        saved_landmarks_map.erase(state_handle);
    }
    bool compute_parent_landmarks(const State &parent_state);
    int cache_hits;
    unsigned int min_cache_hits;
    bool released_memory;
    SavedLandmarksMap *empty;
protected:
    virtual void initialize();
    virtual bool reach_state(const State &parent_state, const Operator &op,
                             const State &state);
    virtual void finished_state(const State &state, int f, bool is_on_frontier);

    virtual void discovered_landmark(const State &state, std::vector<RelaxedOperator *> &landmark, int cost);
    virtual void reset_operator_costs(const State &state);
    virtual int compute_heuristic(const State &state);
public:
    IncrementalLandmarkCutHeuristic(const Options &opts);
    virtual ~IncrementalLandmarkCutHeuristic();
    virtual void free_up_memory(SearchSpace &search_space);
    virtual bool release_memory();
};


#endif
