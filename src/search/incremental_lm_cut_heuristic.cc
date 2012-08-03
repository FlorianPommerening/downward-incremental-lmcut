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
#include <algorithm>

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


void IncrementalLandmarkCutHeuristic::restore_operator_costs(SavedHeuristicOutcome &saved) {
    if (saved.parent_id == -1 || !saved_outcomes[saved.parent_id]) {
        LandmarkCutHeuristic::reset_operator_costs();
    } else {
        restore_operator_costs(*saved_outcomes[saved.parent_id]);
    }
    typedef vector<pair<RelaxedOperator *, int> >::iterator iterator;
    for (iterator it=saved.operator_cost_changes.begin(); it != saved.operator_cost_changes.end(); ++it) {
        it->first->cost += it->second;
    }
}


bool IncrementalLandmarkCutHeuristic::reach_state(const State &parent_state, const Operator &op, const State &state) {
    free_up_memory();
    if (!saved_outcomes[parent_state]) {
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
    saved_outcomes[state] = new SavedHeuristicOutcome(parent_state.get_id());
    saved_outcomes[parent_state]->children_ids.push_back(state.get_id());
    vector<LandmarkPtr> &current_landmarks = saved_outcomes[state]->landmarks;
    vector<LandmarkPtr> &parent_landmarks = saved_outcomes[parent_state]->landmarks;
    vector<pair<RelaxedOperator *, int> > &operator_cost_changes = saved_outcomes[state]->operator_cost_changes;
//    used_memory += sizeof(vector<LandmarkPtr>);
    current_landmarks.reserve(parent_landmarks.size());

    restore_operator_costs(*saved_outcomes[parent_state]);

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
                operator_cost_changes.push_back(make_pair(*it_op, landmark->cost));
                used_memory += sizeof(pair<RelaxedOperator *, int>);
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
    saved_outcomes[state]->landmarks.push_back(createdLandmark);
    used_memory += landmark.size() * sizeof(RelaxedOperator *);
    used_memory += sizeof(Landmark);
    used_memory += sizeof(LandmarkPtr);

    for (vector<RelaxedOperator *>::iterator it_op = landmark.begin(); it_op != landmark.end(); ++it_op) {
        saved_outcomes[state]->operator_cost_changes.push_back(make_pair(*it_op, -cost));
        used_memory += sizeof(pair<RelaxedOperator *, int>);
    }
}

void IncrementalLandmarkCutHeuristic::reset_operator_costs() {
    // in all other states we skip this because we copy the old values in reach_state instead of reseting them.
    // it would be nicer to copy them in this function, but we do not have the parent state and used operator here.
    if (!has_parent_costs) {
        LandmarkCutHeuristic::reset_operator_costs();
    }
}

int IncrementalLandmarkCutHeuristic::compute_heuristic(const State &state) {
    if (!saved_outcomes[state]) {
        saved_outcomes[state] = new SavedHeuristicOutcome(-1);
    }
    if (LandmarkCutHeuristic::compute_heuristic(state) == DEAD_END) {
        remove_meta_information(state.get_id());
        return DEAD_END;
    }

    int result = 0;
    vector<LandmarkPtr> &landmarks = saved_outcomes[state]->landmarks;
    for (vector<LandmarkPtr>::iterator it = landmarks.begin(); it != landmarks.end(); ++it) {
        LandmarkPtr landmark = *it;
        result += landmark->cost;
    }

    saved_outcomes[state]->f = result;
    return result;
}

bool is_null(pair<RelaxedOperator *, int> const & x) {
    return x.second == 0;
}

void IncrementalLandmarkCutHeuristic::remove_meta_information(int state_id) {
    SavedHeuristicOutcome *saved = saved_outcomes[state_id];
    if (!saved) {
        return;
    }
    SavedHeuristicOutcome *parent = 0;
    if (saved->parent_id != -1) {
        parent = saved_outcomes[saved->parent_id];
    }
    used_memory -= saved->operator_cost_changes.size() * sizeof(pair<RelaxedOperator *, int>);
    for (vector<int>::iterator child_it=saved->children_ids.begin(); child_it != saved->children_ids.end(); ++child_it) {
        SavedHeuristicOutcome *child = saved_outcomes[*child_it];
        child->parent_id = saved->parent_id;
        if (parent) {
            parent->children_ids.push_back(*child_it);
        }
        used_memory -= child->operator_cost_changes.size() * sizeof(pair<RelaxedOperator *, int>);
        // merge changes from saved and child. leave out entries that cancel each other out.
        map<RelaxedOperator *, int> changes(child->operator_cost_changes.begin(), child->operator_cost_changes.end());
        for (vector<pair<RelaxedOperator*, int> >::iterator it=saved->operator_cost_changes.begin(); it != saved->operator_cost_changes.end(); ++it) {
            map<RelaxedOperator *, int>::iterator entry = changes.find(it->first);
            if (entry != changes.end()) {
                entry->second += it->second;
            } else {
                changes[it->first] = it->second;
            }
        }
        child->operator_cost_changes.clear();
        std::remove_copy_if(changes.begin(), changes.end(), std::back_inserter(child->operator_cost_changes), is_null);
        used_memory += child->operator_cost_changes.size() * sizeof(pair<RelaxedOperator *, int>);
    }

    for (vector<LandmarkPtr>::iterator it_landmark = saved->landmarks.begin(); it_landmark != saved->landmarks.end(); ++it_landmark) {
        LandmarkPtr &landmark = *it_landmark;
        used_memory -= sizeof(LandmarkPtr);
        if (landmark.unique()) {
            used_memory -= landmark->size() * sizeof(RelaxedOperator *);
            used_memory -= sizeof(Landmark);
        }
    }

    if (parent) {
        parent->children_ids.erase(std::remove(parent->children_ids.begin(), parent->children_ids.end(), state_id), parent->children_ids.end());
    }
    delete saved;
    saved_outcomes[state_id] = 0;
//    used_memory -= sizeof(vector<LandmarkPtr>);
}

void IncrementalLandmarkCutHeuristic::free_up_memory() {
    if (used_memory < memory_limit) {
        return;
    }
    int memory_target = memory_limit; // could also try memory_limit/2, ...
    // remove until in limit or no more nodes scheduled for deletion
    while (used_memory > memory_target && !deletion_schedule.empty()) {
        int i = deletion_schedule.back();
        deletion_schedule.pop_back();
        remove_meta_information(i);
    }
    if (deletion_schedule.empty()) {
        // no more nodes are scheduled but we need to delete more.
        // schedule all current nodes for removal in order of decreasing f value
        cout << "Collecting nodes for removal ... ";
        for (int i = 0; i != saved_outcomes.size(); ++i) {
            if (saved_outcomes[i]) {
                deletion_schedule.push_back(i);
            }
        }
        sort(deletion_schedule.begin(), deletion_schedule.end(), SortByFValue(saved_outcomes));
        cout << "done" << endl;
        // call this again to ensure that enough nodes are deleted
        free_up_memory();
    }
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
