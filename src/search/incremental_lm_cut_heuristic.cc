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

Landmark::Landmark(vector<RelaxedOperator *> &operators, int cost) :
    vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> >(operators.begin(), operators.end()),
    cost(cost) {
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
    free_up_memory();
    if (!saved_outcomes[parent_state]) {
        has_parent_data = false;
        if (reevaluate_parent) {
            compute_heuristic(parent_state);
            // now the information for the parent node has to exist. continue as planned
        } else {
            return true;
        }
    }

    has_parent_data = true;
    bool h_dirty = false;
    saved_outcomes[state] = new SavedHeuristicOutcome();
    operator_cost_changes[state] = new OperatorCostChanges(parent_state.get_id());

    operator_cost_changes[parent_state.get_id()]->open_children++;

    vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> > &current_landmarks = saved_outcomes[state]->landmarks;
    vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> > &parent_landmarks = saved_outcomes[parent_state]->landmarks;

    operator_cost_changes[parent_state]->update_costs(relaxed_operators, operator_cost_changes);

    for (vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> >::iterator it_landmark = parent_landmarks.begin(); it_landmark != parent_landmarks.end(); ++it_landmark) {
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
            for (vector<RelaxedOperator *, MemoryTrackingAllocator<RelaxedOperator *> >::iterator it_op = landmark->begin(); it_op != landmark->end(); ++it_op) {
                (*it_op)->cost += landmark->cost;
                operator_cost_changes[state]->push_back(make_pair(*it_op, landmark->cost));
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

    for (vector<RelaxedOperator *>::iterator it_op = landmark.begin(); it_op != landmark.end(); ++it_op) {
        operator_cost_changes[state]->push_back(make_pair(*it_op, -cost));
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
    if (!saved_outcomes[state]) {
        saved_outcomes[state] = new SavedHeuristicOutcome();
        operator_cost_changes[state] = new OperatorCostChanges(-1);
    }
    if (LandmarkCutHeuristic::compute_heuristic(state) == DEAD_END) {
        remove_meta_information(state.get_id());
        return DEAD_END;
    }

    int result = 0;
    vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> > &landmarks = saved_outcomes[state]->landmarks;
    for (vector<LandmarkPtr, MemoryTrackingAllocator<LandmarkPtr> >::iterator it = landmarks.begin(); it != landmarks.end(); ++it) {
        LandmarkPtr landmark = *it;
        result += landmark->cost;
    }

    saved_outcomes[state]->f = result;
    return result;
}

void IncrementalLandmarkCutHeuristic::remove_meta_information(int state_id) {
    if (!saved_outcomes[state_id]) {
        return;
    }

    delete saved_outcomes[state_id];
    saved_outcomes[state_id] = 0;

    // delete all op cost infos that are no longer needed
    int id = state_id;
    while (id != -1 && !saved_outcomes[id] && operator_cost_changes[id]->open_children == 0) {
        int parent_id = operator_cost_changes[id]->parent_id;
        delete operator_cost_changes[id];
        operator_cost_changes[id] = 0;
        if (parent_id != -1) {
            operator_cost_changes[parent_id]->open_children--;
        }
        id = parent_id;
    }

}

void IncrementalLandmarkCutHeuristic::free_up_memory(bool can_recurse) {
    if (g_memory_tracking_allocated < memory_limit) {
        return;
    }
    int memory_target = memory_limit; // could also try memory_limit/2, ...
    // remove until in limit or no more nodes scheduled for deletion
    while (g_memory_tracking_allocated > memory_target && !deletion_schedule.empty()) {
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
        cout << "done (" << deletion_schedule.size() << " elements scheduled for deletion)" << endl;
        // call this again to ensure that enough nodes are deleted
        if (can_recurse && !deletion_schedule.empty()) {
            free_up_memory(false);
        }
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
