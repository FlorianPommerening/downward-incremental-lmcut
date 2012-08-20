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

Landmark::Landmark(vector<RelaxedOperator *> &operators, int cost) :
    vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> >(operators.begin(), operators.end()),
    cost(cost) {
}

OperatorCostChanges::OperatorCostChanges(StateProxy _parent_proxy) :
    parent_proxy(_parent_proxy),
    open_children(0) {
}

void OperatorCostChanges::update_costs(vector<RelaxedOperator> &operators, PerNodeInfoHashMap &node_infos) {

    if (parent_proxy.state_data != 0) {
        PerNodeInfoHashMap::iterator node_info_it = node_infos.find(parent_proxy);
        assert(node_info_it != node_infos.end());
        OperatorCostChanges &parent = node_info_it->second.cost_changes;
        parent.update_costs(operators, node_infos);
    } else {
        for (vector<RelaxedOperator>::iterator it=operators.begin(); it != operators.end(); ++it) {
            it->cost = it->base_cost;
        }
    }
    for (iterator it=begin(); it != end(); ++it) {
        it->first->cost += it->second;
    }

}

// construction and destruction
IncrementalLandmarkCutHeuristic::IncrementalLandmarkCutHeuristic(const Options &opts)
    : LandmarkCutHeuristic(opts),
      keep_frontier(opts.get<bool>("keep_frontier")),
      reevaluate_parent(opts.get<bool>("reevaluate_parent")),
      memory_limit(opts.get<unsigned int>("memory_limit")),
      has_parent_data(false) {
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
    StateProxy parent_proxy = StateProxy(&parent_state);
    PerNodeInfoHashMap::iterator parent_node_info_it = node_infos.find(parent_proxy);
    if (parent_node_info_it == node_infos.end()) {
        has_parent_data = false;
        if (reevaluate_parent) {
            compute_heuristic(parent_state);
            parent_node_info_it = node_infos.find(parent_proxy);
            assert(parent_node_info_it != node_infos.end());
            // now the information for the parent node has to exist. continue as planned
        } else {
            return true;
        }
    }

    has_parent_data = true;
    bool h_dirty = false;

    StateProxy state_proxy(&state);
    PerNodeInfoHashMap::iterator state_node_info_it = node_infos.find(state_proxy);
    if (state_node_info_it == node_infos.end()) {
        pair<PerNodeInfoHashMap::iterator, bool> insert_result =
            node_infos.insert(make_pair(state_proxy,
                                        PerNodeInfo(SavedLandmarks(),
                                                    OperatorCostChanges(parent_proxy))));
        state_node_info_it = insert_result.first;
    }

    OperatorCostChanges &parent_cost_changes = parent_node_info_it->second.cost_changes;
    OperatorCostChanges &current_cost_changes = state_node_info_it->second.cost_changes;
    SavedLandmarks &parent_landmarks = parent_node_info_it->second.landmarks;
    SavedLandmarks &current_landmarks = state_node_info_it->second.landmarks;

    parent_cost_changes.open_children++;
    parent_cost_changes.update_costs(relaxed_operators, node_infos);


    for (SavedLandmarks::iterator it_landmark = parent_landmarks.begin(); it_landmark != parent_landmarks.end(); ++it_landmark) {
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
        } else {
            for (Landmark::iterator it_op = landmark->begin(); it_op != landmark->end(); ++it_op) {
                (*it_op)->cost += landmark->cost;
                current_cost_changes.push_back(make_pair(*it_op, landmark->cost));
            }
        }
    }
    return h_dirty;
}

void IncrementalLandmarkCutHeuristic::finished_state(const State &state, int /*f*/,
                                                     bool is_on_frontier) {
    StateProxy proxy(&state);
    if (is_on_frontier && keep_frontier) {
        // TODO could also keep only a fixed amount of states (keep lower f values)
        return;
    } else {
        remove_meta_information(proxy);
    }
}

void IncrementalLandmarkCutHeuristic::discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark, int cost) {
    PerNodeInfoHashMap::iterator state_info_it = node_infos.find(StateProxy(&state));
    assert(state_info_it != node_infos.end());
    PerNodeInfo &node_info = state_info_it->second;

    LandmarkPtr createdLandmark(new Landmark(landmark, cost));
    node_info.landmarks.push_back(createdLandmark);
    OperatorCostChanges &cost_changes = node_info.cost_changes;
    for (vector<RelaxedOperator *>::iterator it_op = landmark.begin(); it_op != landmark.end(); ++it_op) {
        cost_changes.push_back(make_pair(*it_op, -cost));
    }
}

void IncrementalLandmarkCutHeuristic::reset_operator_costs() {
    // in all other states we skip this because we copy the old values in reach_state instead of reseting them.
    // it would be nicer to copy them in this function, but we do not have the parent state and used operator here.
    if (!has_parent_data) {
        LandmarkCutHeuristic::reset_operator_costs();
    }
}

int IncrementalLandmarkCutHeuristic::compute_heuristic(const State &state) {
    StateProxy state_proxy(&state);
    PerNodeInfoHashMap::iterator state_node_info_it = node_infos.find(state_proxy);
    if (state_node_info_it == node_infos.end()) {
        pair<PerNodeInfoHashMap::iterator, bool> insert_result =
            node_infos.insert(make_pair(state_proxy,
                                        PerNodeInfo(SavedLandmarks(),
                                                    OperatorCostChanges(StateProxy()))));
        state_node_info_it = insert_result.first;
    }
    if (LandmarkCutHeuristic::compute_heuristic(state) == DEAD_END) {
        remove_meta_information(state_proxy);
        return DEAD_END;
    }

    int result = 0;
    SavedLandmarks &landmarks = state_node_info_it->second.landmarks;
    for (SavedLandmarks::iterator it = landmarks.begin(); it != landmarks.end(); ++it) {
        LandmarkPtr landmark = *it;
        result += landmark->cost;
    }
    return result;
}

void IncrementalLandmarkCutHeuristic::remove_meta_information(StateProxy &state_proxy) {
    PerNodeInfoHashMap::iterator state_node_info_it = node_infos.find(state_proxy);
    if (state_node_info_it == node_infos.end()) {
        return;
    }
    // clearing by swaping with empty vector to ensure capacity reset;
    SavedLandmarks empty;
    state_node_info_it->second.landmarks.swap(empty);

    // delete all op cost infos that are no longer needed
    for(;;) {
        if (!state_node_info_it->second.landmarks.empty()) {
            break;
        }
        OperatorCostChanges &cost_changes = state_node_info_it->second.cost_changes;
        if (cost_changes.open_children > 0) {
            break;
        }
        node_infos.erase(state_node_info_it);
        if (cost_changes.parent_proxy.state_data) {
            state_node_info_it = node_infos.find(cost_changes.parent_proxy);
            state_node_info_it->second.cost_changes.open_children--;
        } else {
            break;
        }
    }
}

void IncrementalLandmarkCutHeuristic::free_up_memory(SearchSpace &search_space) {
    if (g_memory_tracking_allocated < memory_limit) {
        return;
    }
    int n = node_infos.size();
    cout << "Freeing memory by deleting meta information ... ";
    // try to find the nodes with maximal f value
    vector<pair<int, StateProxy> > states_by_f_value;
    states_by_f_value.reserve(n);
    for (PerNodeInfoHashMap::iterator it = node_infos.begin(); it != node_infos.end(); ++it) {
        StateProxy s = it->first;
        SearchNode n = search_space.get_node(State(s.state_data));
        // TODO this could lead to problem with multiple heuristics because n.get_h(); will return the value of only one of the heuristics
        int f = n.get_real_g() + n.get_h();
        pair<int, StateProxy> p = make_pair(f, s);
        states_by_f_value.push_back(p);
    }
    sort(states_by_f_value.begin(), states_by_f_value.end(), ComparePairByFirstComponent<int, StateProxy>());
    for (int i = states_by_f_value.size() - 1; i >= 0 && g_memory_tracking_allocated > memory_limit / 2; --i) {
        StateProxy s = states_by_f_value[i].second;
        remove_meta_information(s);
    }
    cout << "done (removed info for " << n - node_infos.size() << " of " << n << " states)" << endl;
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
