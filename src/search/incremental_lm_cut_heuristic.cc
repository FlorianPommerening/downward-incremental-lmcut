#include "incremental_lm_cut_heuristic.h"

#include "globals.h"
#include "operator.h"
#include "option_parser.h"
#include "plugin.h"
#include "state.h"
#include "state_var_t.h"
#include "rng.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
using namespace std;

Landmark::Landmark(vector<RelaxedOperator *> &operators, int cost) : vector<RelaxedOperator *>(operators), cost(cost) {
}

// construction and destruction
IncrementalLandmarkCutHeuristic::IncrementalLandmarkCutHeuristic(const Options &opts)
    : LandmarkCutHeuristic(opts),
      keep_frontier(opts.get<bool>("keep_frontier")),
      reevaluate_parent(opts.get<bool>("reevaluate_parent")),
      memory_limit(opts.get<unsigned int>("memory_limit")),
      used_memory(0),
      has_parent_costs(false) {
}

IncrementalLandmarkCutHeuristic::~IncrementalLandmarkCutHeuristic() {
}

void IncrementalLandmarkCutHeuristic::initialize() {
    LandmarkCutHeuristic::initialize();

    cout << "Landmarks for node on the search frontier will "
         << (keep_frontier ? "not " : "") << "be removed" << endl;

    cout << "For nodes without saved parent landmarks the parent node will "
         << (reevaluate_parent ? "" : "not ") << "be reevaluated" << endl;

    if (memory_limit == 0) {
        memory_limit = numeric_limits<unsigned long>::max();
        cout << "Using no memory limit" << endl;
    } else {
        // convert MB in byte
        cout << "Memory limit is " << memory_limit * 1024 * 1024 << " bytes (" << memory_limit << " MB). Meta information will be pruned if it takes up more space." << endl;
        memory_limit *= 1024 * 1024;
    }
}


bool IncrementalLandmarkCutHeuristic::reach_state(const State &parent_state, const Operator &op, const State &state) {
    if (saved_outcomes[parent_state].operator_costs.empty()) {
        has_parent_costs = false;
        if (reevaluate_parent) {
            compute_heuristic(parent_state);
            // now the information for the parent node has to exist. continue as planned
        } else {
            return true;
        }
    }

    has_parent_costs = true;
    bool h_dirty = false;
    vector<LandmarkPtr> &current_landmarks = saved_outcomes[state].landmarks;
    vector<int> &parent_operator_costs = saved_outcomes[parent_state].operator_costs;
    vector<LandmarkPtr> &parent_landmarks = saved_outcomes[parent_state].landmarks;
//    used_memory += sizeof(vector<LandmarkPtr>);
    current_landmarks.reserve(parent_landmarks.size());

    for (int i = 0; i < parent_operator_costs.size(); ++i) {
        relaxed_operators[i].cost = parent_operator_costs[i];
    }

    for (vector<LandmarkPtr>::iterator it_landmark = parent_landmarks.begin(); it_landmark != parent_landmarks.end(); ++it_landmark) {
        LandmarkPtr landmark = *it_landmark;
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
            used_memory += sizeof(LandmarkPtr);
        } else {
            for (vector<RelaxedOperator *>::iterator it_op = landmark->begin(); it_op != landmark->end(); ++it_op) {
                (*it_op)->cost += landmark->cost;
            }
        }
    }
    return h_dirty;
}

void IncrementalLandmarkCutHeuristic::finished_state(const State &state, int /*f*/,
                                                     bool is_on_frontier) {
    if (is_on_frontier && keep_frontier) {
        // TODO could also keep only a fixed amount of states (keep lower f values)
        return;
    } else {
        remove_meta_information(state.get_id());
    }
}

void IncrementalLandmarkCutHeuristic::discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost) {
    LandmarkPtr createdLandmark(new Landmark(landmark, cost));
    saved_outcomes[state].landmarks.push_back(createdLandmark);
    used_memory += landmark.size() * sizeof(RelaxedOperator *);
    used_memory += sizeof(Landmark);
    used_memory += sizeof(LandmarkPtr);
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
        remove_meta_information(state.get_id());
        return DEAD_END;
    }

    // remember operator costs
    vector<int> &operator_costs = saved_outcomes[state].operator_costs;
    operator_costs.resize(relaxed_operators.size());
    for (int i = 0; i < relaxed_operators.size(); ++i) {
        operator_costs[i] = relaxed_operators[i].cost;
    }
    used_memory += operator_costs.size() * sizeof(int);
//    used_memory += sizeof(vector<int>);

    int result = 0;
    vector<LandmarkPtr> &landmarks = saved_outcomes[state].landmarks;
    for (vector<LandmarkPtr>::iterator it = landmarks.begin(); it != landmarks.end(); ++it) {
        LandmarkPtr landmark = *it;
        result += landmark->cost;
    }
    return result;
}

void IncrementalLandmarkCutHeuristic::remove_meta_information(int state_id) {
    vector<int> &operator_costs = saved_outcomes[state_id].operator_costs;
    used_memory -= operator_costs.size() * sizeof(int);
    operator_costs.clear();
//        used_memory -= sizeof(vector<int>);

    vector<LandmarkPtr> &landmarks = saved_outcomes[state_id].landmarks;
    for (vector<LandmarkPtr>::iterator it_landmark = landmarks.begin(); it_landmark != landmarks.end(); ++it_landmark) {
        LandmarkPtr &landmark = *it_landmark;
        used_memory -= sizeof(LandmarkPtr);
        if (landmark.unique()) {
            used_memory -= landmark->size() * sizeof(RelaxedOperator *);
            used_memory -= sizeof(Landmark);
        }
    }
    landmarks.clear();
//    used_memory -= sizeof(vector<LandmarkPtr>);
}

void IncrementalLandmarkCutHeuristic::free_up_memory(SearchSpace &/*search_space*/) {
    if (used_memory < memory_limit) {
        return;
    }
    int n = saved_outcomes.size();
    int deleted = 0;
    cout << "Freeing memory by deleting meta information ... ";
    int memory_target = memory_limit / 2;
    while (used_memory > memory_target) {
        int i = g_rng.next(n);
        while (saved_outcomes[i % n].operator_costs.empty()) {
            i++;
        }
        remove_meta_information(i);
        deleted++;
    }
    cout << "done (removed info for " << deleted << " of " << n << " states)" << endl;
}


static ScalarEvaluator *_parse(OptionParser &parser) {
    parser.add_option<unsigned int>("memory_limit",  0, "Maximum amount of memory (in MB) used for storing meta information. Use 0 (default) for no limit.");
    parser.add_option<bool>("keep_frontier",     false, "Landmarks for nodes on the search frontier are not removed if the node is closed.");
    parser.add_option<bool>("reevaluate_parent", false, "If a node is encountered where the parent node has no saved landmarks, the heuristic for the parent is recomputed. This is based on the assumption that siblings of this node will be expanded soon and will profit from the calculated parent info");
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new IncrementalLandmarkCutHeuristic(opts);
}


static Plugin<ScalarEvaluator> _plugin("incremental_lmcut", _parse);
