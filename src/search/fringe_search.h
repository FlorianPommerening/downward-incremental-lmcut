#ifndef FRINGE_SEARCH_H
#define FRINGE_SEARCH_H

#include <list>

#include "evaluator.h"
#include "search_engine.h"
#include "search_space.h"
#include "state_var_t.h"
#include "timer.h"

class Heuristic;
class Operator;
class ScalarEvaluator;
class Options;

typedef std::list<state_var_t *> Fringe;

class FringeSearch : public SearchEngine {
    Heuristic *heuristic;
    Fringe fringe;
    int f_limit;
protected:
    int step();
    virtual void initialize();
    virtual void statistics() const;

public:
    FringeSearch(const Options &opts);
};

#endif
