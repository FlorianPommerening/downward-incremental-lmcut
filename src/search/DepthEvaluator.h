#ifndef DEPTHEVALUATOR_H
#define DEPTHEVALUATOR_H

#include "state.h"
#include "heuristic.h"

class DepthEvaluator : public Heuristic {
public:
    DepthEvaluator(Options &opts): Heuristic(opts) {
    }

    virtual int compute_heuristic(const State &state){
        return state.get_id();//100-state.depth;
    }
};

#endif // DEPTHEVALUATOR_H
