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
    in_initial_state = true;
}

IncrementalLandmarkCutHeuristic::~IncrementalLandmarkCutHeuristic() {
    for (vector<Landmark *>::iterator it = all_landmarks.begin(); it != all_landmarks.end(); ++it) {
        delete *it;
    }
}

bool IncrementalLandmarkCutHeuristic::reach_state(const State &parent_state, const Operator &op, const State &state) {
    in_initial_state = false;
    bool h_dirty = false;
    vector<int> &parent_operator_costs = node_operator_costs[StateProxy(&parent_state)];
    vector<Landmark *> &parent_landmarks = node_landmarks[StateProxy(&parent_state)];
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
    if (in_initial_state) {
        LandmarkCutHeuristic::reset_operator_costs();
    }

}

int IncrementalLandmarkCutHeuristic::compute_heuristic(const State &state) {
    LandmarkCutHeuristic::compute_heuristic(state);

    // remember operator costs
    node_operator_costs[StateProxy(&state)] = vector<int>(relaxed_operators.size());
    vector<int> &operator_costs = node_operator_costs[StateProxy(&state)];
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


static ScalarEvaluator *_parse(OptionParser &parser) {
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new IncrementalLandmarkCutHeuristic(opts);
}


static Plugin<ScalarEvaluator> _plugin("incremental_lmcut", _parse);
