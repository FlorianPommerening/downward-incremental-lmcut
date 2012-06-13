#include "incremental_lm_cut_heuristic.h"

#include "globals.h"
#include "operator.h"
#include "option_parser.h"
#include "plugin.h"
#include "state.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
using namespace std;

Landmark::Landmark(vector<RelaxedOperator *> &operators, int cost): vector<RelaxedOperator *>(operators), cost(cost) {
}

// construction and destruction
IncrementalLandmarkCutHeuristic::IncrementalLandmarkCutHeuristic(const Options &opts)
    : LandmarkCutHeuristic(opts) {
    has_parent_costs = false;
}

IncrementalLandmarkCutHeuristic::~IncrementalLandmarkCutHeuristic() {
    for (vector<Landmark *>::iterator it = all_landmarks.begin(); it != all_landmarks.end(); ++it) {
        delete *it;
    }
}

void IncrementalLandmarkCutHeuristic::initialize() {
    LandmarkCutHeuristic::initialize();

    //HACK hardcoded for testing
    // ~1GB for the meta data assuming:
    //     1 int for each operator in op costs
    // + 1.5 pointers for each operator in landmarks (for operators occuring in more than one LM)
    // + 0.5 pointers for each operator in landmarks (for vectors of individual LMs)
    max_entries = 1024 * 1024 * 1024  / (4* 3 * relaxed_operators.size());
    cout << "Setting max_entries to " << max_entries << endl;
}


bool IncrementalLandmarkCutHeuristic::reach_state(const State &parent_state, const Operator &op, const State &state) {
    StateProxy parent_proxy = StateProxy(&parent_state);
    if (node_operator_costs.find(parent_proxy) == node_operator_costs.end()) {
        has_parent_costs = false;
        assert(node_landmarks.find(parent_proxy) == node_landmarks.end());
        return true;
    }

    has_parent_costs = true;
    bool h_dirty = false;
    vector<int> &parent_operator_costs = node_operator_costs[parent_proxy];
    vector<Landmark *> &parent_landmarks = node_landmarks[parent_proxy];
    vector<Landmark *> &current_landmarks = node_landmarks[StateProxy(&state)];
    current_landmarks.reserve(parent_landmarks.size());

    for (int i=0; i < parent_operator_costs.size(); ++i) {
        relaxed_operators[i].cost = parent_operator_costs[i];
    }

    for (vector<Landmark *>::iterator it_landmark = parent_landmarks.begin(); it_landmark != parent_landmarks.end(); ++it_landmark) {
        Landmark *landmark = *it_landmark;
        bool copy_landmark = true;
        for (Landmark::iterator it_op = landmark->begin(); it_op != landmark->end(); ++it_op) {
            const Operator *base_operator = (*it_op)->op;
            if (base_operator == &op) {
                copy_landmark = false;
                h_dirty = true;
                break;
            }
        }
        if (copy_landmark) {
            current_landmarks.push_back(landmark);
        }
        else {
            for (vector<RelaxedOperator *>::iterator it_op = landmark->begin(); it_op != landmark->end(); ++it_op) {
                (*it_op)->cost += landmark->cost;
            }
        }
    }
    return h_dirty;
}

void IncrementalLandmarkCutHeuristic::discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost) {
    Landmark *createdLandmark = new Landmark(landmark, cost);
    all_landmarks.push_back(createdLandmark);
    node_landmarks[StateProxy(&state)].push_back(createdLandmark);
}

void IncrementalLandmarkCutHeuristic::reset_operator_costs() {
    // in all other states we skip this because we copy the old values in reach_state instead of reseting them.
    // it would be nicer to copy them in this function, but we do not have the parent state and used operator here.
    if (!has_parent_costs) {
        LandmarkCutHeuristic::reset_operator_costs();
    }

}

int IncrementalLandmarkCutHeuristic::compute_heuristic(const State &state) {
    if (LandmarkCutHeuristic::compute_heuristic(state) == DEAD_END) {
        return DEAD_END;
    }

    // remember operator costs
    vector<int> &operator_costs = node_operator_costs[StateProxy(&state)];
    operator_costs.resize(relaxed_operators.size());
    for (int i=0; i < relaxed_operators.size(); ++i) {
        operator_costs[i] = relaxed_operators[i].cost;
    }

    int result = 0;
    vector<Landmark *> &landmarks = node_landmarks[StateProxy(&state)];
    for (vector<Landmark *>::iterator it = landmarks.begin(); it != landmarks.end(); ++it) {
        Landmark *landmark = *it;
        result += landmark->cost;
    }
    return result;
}

void IncrementalLandmarkCutHeuristic::free_up_memory(SearchSpace &search_space) {
    if (node_operator_costs.size() < max_entries) {
        return;
    }
    cout << "Freeing up memory (" << node_operator_costs.size() << " nodes with meta info, " << max_entries << " allowed)"  << endl;
    // try to find the nodes with maximal f value
    vector<pair<int, StateProxy> > states_by_f_value;
    states_by_f_value.reserve(node_operator_costs.size());
    for(__gnu_cxx::hash_map<StateProxy, vector<int> >::iterator it = node_operator_costs.begin(); it != node_operator_costs.end(); ++it) {
        StateProxy s = it->first;
        SearchNode n = search_space.get_node(State(s.state_data));
        int f = n.get_real_g() + n.get_h();
        pair<int, StateProxy> p = make_pair(f, s);
        states_by_f_value.push_back(p);
    }
    sort(states_by_f_value.begin(), states_by_f_value.end(), ComparePairByFirstComponent<int,StateProxy>());
    for(int i = states_by_f_value.size()-1; i >= max_entries /2; --i) {
        StateProxy s = states_by_f_value[i].second;
        node_operator_costs.erase(s);
        node_landmarks.erase(s);
    }
    cout << "Now there are " << node_operator_costs.size() << " nodes left." << endl;
}


static ScalarEvaluator *_parse(OptionParser &parser) {
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new IncrementalLandmarkCutHeuristic(opts);
}


static Plugin<ScalarEvaluator> _plugin("incremental_lmcut", _parse);
