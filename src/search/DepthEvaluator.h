#ifndef DEPTHEVALUATOR_H
#define DEPTHEVALUATOR_H

#include "state.h"
#include "heuristic.h"

class DepthEvaluator : public Heuristic {
public:
    DepthEvaluator(Options &opts): Heuristic(opts) {
    }

    virtual int compute_heuristic(const State &state){
        return 100-state.depth;
    }
};

class IDEvaluator : public Heuristic {
public:
    IDEvaluator(Options &opts): Heuristic(opts) {
    }

    virtual int compute_heuristic(const State &state){
        return state.get_id();
    }
};
#endif // DEPTHEVALUATOR_H
