#ifndef INCREMENTAL_LM_CUT_HEURISTIC_H
#define INCREMENTAL_LM_CUT_HEURISTIC_H

#include "lm_cut_heuristic.h"
#include "state_proxy.h"


// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark: public vector<RelaxedOperator *> {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

class IncrementalLandmarkCutHeuristic : public LandmarkCutHeuristic {
private:
    vector<Landmark *> all_landmarks;
    __gnu_cxx::hash_map<StateProxy, vector<Landmark *> > node_landmarks;
    __gnu_cxx::hash_map<StateProxy, vector<int> > node_operator_costs;
   bool in_initial_state;
protected:
    virtual bool reach_state(const State &parent_state, const Operator &op,
                             const State &state);
    virtual void discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost);
    virtual void reset_operator_costs();
    virtual int compute_heuristic(const State &state);
public:
    IncrementalLandmarkCutHeuristic(const Options &opts);
    virtual ~IncrementalLandmarkCutHeuristic();
};


#endif
