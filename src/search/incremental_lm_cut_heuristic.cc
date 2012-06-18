#include "incremental_lm_cut_heuristic.h"

#include "globals.h"
#include "operator.h"
#include "option_parser.h"
#include "plugin.h"
#include "state.h"
#include "state_var_t.h"

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
      memory_limit(opts.get<unsigned int>("memory_limit")),
      used_memory(0),
      has_parent_costs(false) {
}

IncrementalLandmarkCutHeuristic::~IncrementalLandmarkCutHeuristic() {
}

void IncrementalLandmarkCutHeuristic::initialize() {
    LandmarkCutHeuristic::initialize();

    if (memory_limit == 0) {
        memory_limit = numeric_limits<int>::max();
        cout << "Using no memory limit" << endl;
    } else {
        // convert MB in byte
        cout << "Memory limit is " << memory_limit * 1024 * 1024 << " bytes (" << memory_limit << " MB). Meta information will be pruned if it takes up more space." << endl;
        memory_limit *= 1024 * 1024;
    }
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
    vector<LandmarkPtr> &parent_landmarks = node_landmarks[parent_proxy];
    vector<LandmarkPtr> &current_landmarks = node_landmarks[StateProxy(&state)];
    used_memory += sizeof(vector<LandmarkPtr>);
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

void IncrementalLandmarkCutHeuristic::discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost) {
    LandmarkPtr createdLandmark(new Landmark(landmark, cost));
    node_landmarks[StateProxy(&state)].push_back(createdLandmark);
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
    StateProxy state_proxy(&state);
    if (LandmarkCutHeuristic::compute_heuristic(state) == DEAD_END) {
        remove_meta_information(state_proxy);
        return DEAD_END;
    }

    // remember operator costs
    vector<int> &operator_costs = node_operator_costs[state_proxy];
    operator_costs.resize(relaxed_operators.size());
    for (int i = 0; i < relaxed_operators.size(); ++i) {
        operator_costs[i] = relaxed_operators[i].cost;
    }
    used_memory += operator_costs.size() * sizeof(int);
    used_memory += sizeof(vector<int>);

    int result = 0;
    vector<LandmarkPtr> &landmarks = node_landmarks[state_proxy];
    for (vector<LandmarkPtr>::iterator it = landmarks.begin(); it != landmarks.end(); ++it) {
        LandmarkPtr landmark = *it;
        result += landmark->cost;
    }
    return result;
}

void IncrementalLandmarkCutHeuristic::remove_meta_information(StateProxy &state_proxy) {
    __gnu_cxx::hash_map<StateProxy, vector<int> >::iterator it_op_costs = node_operator_costs.find(state_proxy);
    if (it_op_costs != node_operator_costs.end()) {
        used_memory -= it_op_costs->second.size() * sizeof(int);
        node_operator_costs.erase(it_op_costs);
        used_memory -= sizeof(vector<int>);
    }


    __gnu_cxx::hash_map<StateProxy, vector<LandmarkPtr> >::iterator it_landmarks = node_landmarks.find(state_proxy);
    if (it_landmarks != node_landmarks.end()) {
        vector<LandmarkPtr> &landmarks = it_landmarks->second;
        for (vector<LandmarkPtr>::iterator it_landmark = landmarks.begin(); it_landmark != landmarks.end(); ++it_landmark) {
            LandmarkPtr &landmark = *it_landmark;
            used_memory -= sizeof(LandmarkPtr);
            if (landmark.unique()) {
                used_memory -= landmark->size() * sizeof(RelaxedOperator *);
                used_memory -= sizeof(Landmark);
            }
        }
        node_landmarks.erase(it_landmarks);
        used_memory -= sizeof(vector<LandmarkPtr>);
    }
}

void IncrementalLandmarkCutHeuristic::free_up_memory(SearchSpace &search_space) {
    if (used_memory < memory_limit) {
        return;
    }
    int n = node_operator_costs.size();
    cout << "Freeing memory by deleting meta information ... ";
    // try to find the nodes with maximal f value
    vector<pair<int, StateProxy> > states_by_f_value;
    states_by_f_value.reserve(n);
    for (__gnu_cxx::hash_map<StateProxy, vector<int> >::iterator it = node_operator_costs.begin(); it != node_operator_costs.end(); ++it) {
        StateProxy s = it->first;
        SearchNode n = search_space.get_node(State(s.state_data));
        // TODO this could lead to problem with multiple heuristics because n.get_h(); will return the value of only one of the heuristics
        int f = n.get_real_g() + n.get_h();
        pair<int, StateProxy> p = make_pair(f, s);
        states_by_f_value.push_back(p);
    }
    sort(states_by_f_value.begin(), states_by_f_value.end(), ComparePairByFirstComponent<int, StateProxy>());
    for (int i = states_by_f_value.size() - 1; i >= 0 && used_memory > memory_limit / 2; --i) {
        StateProxy s = states_by_f_value[i].second;
        remove_meta_information(s);
    }
    cout << "done (removed info for " << n - node_operator_costs.size() << " of " << n << " states)" << endl;
}


static ScalarEvaluator *_parse(OptionParser &parser) {
    parser.add_option<unsigned int>("memory_limit", 0, "Maximum amount of memory (in MB) used for storing meta information. Use 0 (default) for no limit");
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new IncrementalLandmarkCutHeuristic(opts);
}


static Plugin<ScalarEvaluator> _plugin("incremental_lmcut", _parse);
