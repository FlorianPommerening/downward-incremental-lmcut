#ifndef INCREMENTAL_LM_CUT_HEURISTIC_H
#define INCREMENTAL_LM_CUT_HEURISTIC_H

#include "lm_cut_heuristic.h"
#include "state_proxy.h"
#include "search_space.h"
#include "memory_tracking_allocator.h"

#include <boost/shared_ptr.hpp>
#include <ext/hash_map>

// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark : public vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> > {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

struct OperatorCostChanges;

typedef boost::shared_ptr<Landmark> LandmarkPtr;
typedef vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> > SavedLandmarks;
typedef __gnu_cxx::hash_map<StateProxy,
                            SavedLandmarks,
                            __gnu_cxx::hash<StateProxy>,
                            equal_to<StateProxy>,
                            MemoryTrackingAllocator<std::pair<const StateProxy,
                                                              SavedLandmarks> > >
        SavedLandmarksHashMap;
typedef __gnu_cxx::hash_map<StateProxy,
                            OperatorCostChanges*,
                            __gnu_cxx::hash<StateProxy>,
                            equal_to<StateProxy>,
                            MemoryTrackingAllocator<std::pair<StateProxy,
                                                              OperatorCostChanges*> > >
        OperatorCostChangesHashMap;



struct OperatorCostChanges : vector<pair<RelaxedOperator *, int>, MemoryTrackingAllocator<pair<RelaxedOperator *, int> > > {
    OperatorCostChanges(StateProxy _parent_proxy) :
        parent_proxy(_parent_proxy),
        open_children(0) {
    }

    StateProxy parent_proxy;
    int open_children;

    void update_costs(vector<RelaxedOperator> &operators, OperatorCostChangesHashMap &changes) {
        OperatorCostChanges *parent = 0;
        if (parent_proxy.state_data) {
            parent = changes[parent_proxy];
        }
        if (parent) {
            parent->update_costs(operators, changes);
        } else {
            for (vector<RelaxedOperator>::iterator it=operators.begin(); it != operators.end(); ++it) {
                it->cost = it->base_cost;
            }
        }
        for (iterator it=begin(); it != end(); ++it) {
            it->first->cost += it->second;
        }

    }
};


template<class T1, class T2>
struct ComparePairByFirstComponent {
    bool operator()(pair<T1, T2> const &lhs, pair<T1, T2> const &rhs) {
        return lhs.first < rhs.first;
    }
};

class IncrementalLandmarkCutHeuristic : public LandmarkCutHeuristic {
private:
    bool keep_frontier;
    bool reevaluate_parent;
    unsigned long memory_limit;
    SavedLandmarksHashMap node_landmarks;
    OperatorCostChangesHashMap operator_cost_changes;
    bool has_parent_costs;
    void remove_meta_information(StateProxy &state_proxy);
protected:
    virtual void initialize();
    virtual bool reach_state(const State &parent_state, const Operator &op,
                             const State &state);
    virtual void finished_state(const State &state, int f, bool is_on_frontier);

    virtual void discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost);
    virtual void reset_operator_costs();
    virtual int compute_heuristic(const State &state);
public:
    IncrementalLandmarkCutHeuristic(const Options &opts);
    virtual ~IncrementalLandmarkCutHeuristic();
    virtual void free_up_memory(SearchSpace &search_space);
};


#endif
