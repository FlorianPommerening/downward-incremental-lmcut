#ifndef INCREMENTAL_LM_CUT_HEURISTIC_H
#define INCREMENTAL_LM_CUT_HEURISTIC_H

#include "lm_cut_heuristic.h"
#include "state_proxy.h"
#include "search_space.h"

#include <boost/shared_ptr.hpp>


// wraps vector<RelaxedOperator *> to add costs
// adding or removing operators after creation does *not* update costs (not implemented)
class Landmark : public vector<RelaxedOperator *> {
public:
    int cost;
    Landmark(vector<RelaxedOperator *> &operators, int cost);
};

typedef boost::shared_ptr<Landmark> LandmarkPtr;

template<class T1, class T2>
struct ComparePairByFirstComponent {
    bool operator()(pair<T1, T2> const &lhs, pair<T1, T2> const &rhs) {
        return lhs.first < rhs.first;
    }
};

class IncrementalLandmarkCutHeuristic : public LandmarkCutHeuristic {
private:
    unsigned long memory_limit;
    unsigned long used_memory;
    __gnu_cxx::hash_map<StateProxy, vector<LandmarkPtr> > node_landmarks;
    __gnu_cxx::hash_map<StateProxy, vector<int> > node_operator_costs;
    bool has_parent_costs;
    void remove_meta_information(StateProxy &state_proxy);
protected:
    virtual void initialize();
    virtual bool reach_state(const State &parent_state, const Operator &op,
                             const State &state);
    virtual void discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost);
    virtual void reset_operator_costs();
    virtual int compute_heuristic(const State &state);
public:
    IncrementalLandmarkCutHeuristic(const Options &opts);
    virtual ~IncrementalLandmarkCutHeuristic();
    virtual void free_up_memory(SearchSpace &search_space);
};


#endif
