#ifndef INCREMENTAL_LM_CUT_HEURISTIC_H
#define INCREMENTAL_LM_CUT_HEURISTIC_H

#include "lm_cut_heuristic.h"
#include "state_proxy.h"
#include "search_space.h"
#include "per_state_information.h"

#include <boost/shared_ptr.hpp>


// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark : public vector<RelaxedOperator *> {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

typedef boost::shared_ptr<Landmark> LandmarkPtr;

struct SavedHeuristicOutcome {
    vector<LandmarkPtr> landmarks;
    vector<int> operator_costs;
};

class IncrementalLandmarkCutHeuristic : public LandmarkCutHeuristic {
private:
    bool keep_frontier;
    bool reevaluate_parent;
    unsigned long memory_limit;
    unsigned long used_memory;
    PerStateInformation<SavedHeuristicOutcome> saved_outcomes;
    bool has_parent_costs;
    void remove_meta_information(int state_id);
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
