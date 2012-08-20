#ifndef IDA_STAR_SEARCH_H
#define IDA_STAR_SEARCH_H

#include <vector>

#include "open_lists/open_list.h"
#include "search_engine.h"
#include "search_space.h"
#include "state.h"
#include "timer.h"
#include "evaluator.h"
#include "search_progress.h"

class Heuristic;
class Operator;
class ScalarEvaluator;
class Options;

class OpenStackEntry {
private:
    std::vector<const Operator *> applicable_operators;
    int last_expanded_operator;

public:
    OpenStackEntry(state_var_t *_state_buffer);
    state_var_t *state_buffer;
    const Operator *next_applicable_operator();
    void reset();
};

class IDAStarSearch : public SearchEngine {
    vector<OpenStackEntry> open_stack;
    bool remember_pruned_nodes;
    vector<OpenStackEntry> next_layer_open_stack;
    bool lookup_known_f_values;
    void backtrack();

protected:
    int current_bound;
    int layer;
    int step();

    Heuristic * heuristic;
    virtual void initialize();
    int restart_with_next_bound(int _current_bound);
public:
    IDAStarSearch(const Options &opts);
    void statistics() const;
};

#endif
