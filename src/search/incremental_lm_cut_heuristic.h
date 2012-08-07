#ifndef INCREMENTAL_LM_CUT_HEURISTIC_H
#define INCREMENTAL_LM_CUT_HEURISTIC_H

#include "lm_cut_heuristic.h"
#include "state_proxy.h"
#include "search_space.h"
#include "per_state_information.h"
#include "memory_tracking_allocator.h"

#include <boost/shared_ptr.hpp>


// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark : public vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> > {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

typedef boost::shared_ptr<Landmark> LandmarkPtr;

struct SavedHeuristicOutcome {
    vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> > landmarks;
    int f;
};

struct OperatorCostChanges : vector<pair<RelaxedOperator *, int>, MemoryTrackingAllocator<pair<RelaxedOperator *, int> > > {
    OperatorCostChanges(int _parent_id) :
        parent_id(_parent_id),
        open_children(0) {
    }

    int parent_id;
    int open_children;

    void update_costs(vector<RelaxedOperator> &operators, PerStateInformation<OperatorCostChanges *> &changes) {
        OperatorCostChanges *parent = 0;
        if (parent_id != -1) {
            parent = changes[parent_id];
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

struct SortByFValue {
    PerStateInformation<SavedHeuristicOutcome *> &saved_outcomes;
    SortByFValue(PerStateInformation<SavedHeuristicOutcome *> &_saved_outcomes) :
        saved_outcomes(_saved_outcomes) {}

    bool operator()(int const &lhs, int const &rhs) {
        return saved_outcomes[lhs]->f < saved_outcomes[rhs]->f;
    }
};

class IncrementalLandmarkCutHeuristic : public LandmarkCutHeuristic {
private:
    bool keep_frontier;
    bool reevaluate_parent;
    unsigned long memory_limit;

    PerStateInformation<SavedHeuristicOutcome *> saved_outcomes;
    PerStateInformation<OperatorCostChanges *> operator_cost_changes;
    bool has_parent_data;

    vector<int> deletion_schedule;
    void remove_meta_information(int state_id);
    virtual void free_up_memory(bool can_recurse=true);
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
};


#endif
