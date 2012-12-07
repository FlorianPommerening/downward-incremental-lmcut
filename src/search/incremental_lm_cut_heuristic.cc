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

// construction and destruction
IncrementalLandmarkCutHeuristic::IncrementalLandmarkCutHeuristic(const Options &opts)
    : LandmarkCutHeuristic(opts),
      local_computation_on(opts.get<bool>("local")),
      reevaluating_parent(false),
      keep_frontier(opts.get<bool>("keep_frontier")),
      reevaluate_parent(opts.get<bool>("reevaluate_parent")),
      memory_limit(opts.get<unsigned int>("memory_limit")),
      current_state_proxy(g_initial_state),
      current_landmarks_cost(0),
      current_parent_state_proxy(g_initial_state) {
}

IncrementalLandmarkCutHeuristic::~IncrementalLandmarkCutHeuristic() {
}

void IncrementalLandmarkCutHeuristic::initialize() {
    LandmarkCutHeuristic::initialize();

    cout << "Landmarks for node on the search frontier will "
         << (keep_frontier ? "not " : "") << "be removed" << endl;

    cout << "For nodes without saved parent landmarks the parent node will "
         << (reevaluate_parent ? "" : "not ") << "be reevaluated" << endl;

    if (local_computation_on) {
        cout << "Incremental computation will only be used locally" << endl;
    }

    if (memory_limit == 0) {
        memory_limit = numeric_limits<unsigned long>::max();
        cout << "Using no memory limit" << endl;
    } else {
        // convert MB in byte
        cout << "Memory limit is " << memory_limit * 1024 * 1024 << " bytes (" << memory_limit << " MB). Meta information will be pruned if it takes up more space." << endl;
        memory_limit *= 1024 * 1024;
    }
    CompressedLandmark::set_operators(relaxed_operators);
}

void IncrementalLandmarkCutHeuristic::reset_operator_costs(const State &state) {
    // Redo cost changes of current landmarks (if there are no current landmarks,
    // this will just reset all costs to base costs)
    LandmarkCutHeuristic::reset_operator_costs(state);
    assert(current_decompressed_landmarks.empty() || StateProxy(state) == current_state);
    for (vector<Landmark>::iterator it_landmark = current_decompressed_landmarks.begin(); it_landmark != current_decompressed_landmarks.end(); ++it_landmark) {
        Landmark &landmark = *it_landmark;
        for (Landmark::iterator it_op = landmark.begin(); it_op != landmark.end(); ++it_op) {
            (*it_op)->cost -= landmark.cost;
        }
    }
}

bool IncrementalLandmarkCutHeuristic::get_parent_landmarks(const State &parent_state) {
    StateProxy parent_proxy = StateProxy(&parent_state);
    if (parent_proxy == current_parent_state_proxy) {
        return true;
    }
    parent_landmarks.clear();
    current_parent_state_proxy = parent_proxy;

    SavedLandmarksMap::iterator parent_node_info_it = saved_landmarks_map.find(parent_proxy);
    if (parent_node_info_it != saved_landmarks_map.end()) {
        SavedLandmarks &compresssed_landmarks = parent_node_info_it->second;
        for (size_t i = 0; i < compresssed_landmarks.size(); ++i) {
            parent_landmarks.push_back(make_pair(compresssed_landmarks[i].decompress(), compresssed_landmarks[i]));
        }
        return true;
    } else {
        if (!reevaluate_parent) {
            return false;
        }
        reevaluating_parent = true;
        compute_heuristic(parent_state);
        reevaluating_parent = false;
        return true;
    }
}

bool IncrementalLandmarkCutHeuristic::reach_state(const State &parent_state, const Operator &op, const State &state) {
    current_decompressed_landmarks.clear();
    current_state_proxy = StateProxy(&state);
    current_landmarks_cost = 0;
    bool h_dirty = false;

    if (!get_parent_landmarks(parent_state)) {
        return h_dirty;
    }

    SavedLandmarks *current_landmarks = 0;
    if (!local_computation_on) {
        current_landmarks = &saved_landmarks_map[current_state_proxy];
    }

    // Copy all landmarks that do not mention op
    for (size_t i = 0; i < parent_landmarks.size(); ++i) {
        Landmark landmark = parent_landmarks[i].first;
        bool copy_landmark = true;
        for (Landmark::iterator it_op = landmark.begin(); it_op != landmark.end(); ++it_op) {
            const Operator *base_operator = (*it_op)->op;
            if (base_operator == &op) {
                copy_landmark = false;
                h_dirty = true;
                break;
            }
        }
        if (copy_landmark) {
            current_landmarks_cost += landmark.cost;
            current_decompressed_landmarks.push_back(landmark);
            if (current_landmarks) {
                current_landmarks->push_back(parent_landmarks[i].second);
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
        StateProxy proxy(&state);
        remove_meta_information(proxy);
    }
}

void IncrementalLandmarkCutHeuristic::discovered_landmark(const State &state, vector<RelaxedOperator *> &landmark_, int cost) {
    if (reevaluating_parent) {
        Landmark landmark(landmark_, cost);
        CompressedLandmark compressed_landmark = CompressedLandmark::EMPTY;
        // avoid compression when using local computation
        if (!local_computation_on) {
            compressed_landmark = CompressedLandmark(landmark_, cost);
        }
        parent_landmarks.push_back(make_pair(landmark, compressed_landmark));
    } else {
        current_landmarks_cost += cost;
        if (!local_computation_on) {
            saved_landmarks_map[StateProxy(&state)].push_back(CompressedLandmark(landmark_, cost));
        }
    }
}

int IncrementalLandmarkCutHeuristic::compute_heuristic(const State &state) {
    StateProxy state_proxy(&state);
    if (LandmarkCutHeuristic::compute_heuristic(state) == DEAD_END) {
        remove_meta_information(state_proxy);
        return DEAD_END;
    }
    return current_landmarks_cost;
}

void IncrementalLandmarkCutHeuristic::free_up_memory(SearchSpace &/*search_space*/) {
    if (local_computation_on || g_memory_tracking_allocated < memory_limit) {
        return;
    }
    int n = saved_landmarks_map.size();
    saved_landmarks_map.clear();
    if (n > 1) {
        cout << "Freeing memory by deleting meta information for all " << n << " states." << endl;
    } else {
        cout << "Memory bounds allows to store only one landmark. Switching to h^iLM-cut_local." << endl;
        local_computation_on = true;
        reevaluate_parent = true;
    }
}


static ScalarEvaluator *_parse(OptionParser &parser) {
    parser.add_option<unsigned int>("memory_limit",  0, "Maximum amount of memory (in MB) used for storing meta information. Use 0 (default) for no limit.");
    parser.add_option<bool>("keep_frontier",     false, "Landmarks for nodes on the search frontier are not removed if the node is closed.");
    parser.add_option<bool>("reevaluate_parent", false, "If a node is encountered where the parent node has no saved landmarks, the heuristic for the parent is recomputed. This is based on the assumption that siblings of this node will be expanded soon and will profit from the calculated parent info");
    parser.add_option<bool>("local", false, "Only use incremental computation locally, i.e. reevaluate the parent and then incrementally compute the child nodes. Automatically enables reevaluate_parent");
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (opts.get<bool>("local")) {
        opts.set<bool>("reevaluate_parent", true);
    }

    if (parser.dry_run())
        return 0;
    else
        return new IncrementalLandmarkCutHeuristic(opts);
}


static Plugin<ScalarEvaluator> _plugin("incremental_lmcut", _parse);
