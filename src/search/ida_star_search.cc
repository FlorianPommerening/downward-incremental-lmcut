#include "ida_star_search.h"

#include "globals.h"
#include "heuristic.h"
#include "option_parser.h"
#include "successor_generator.h"
#include "plugin.h"

#include <cassert>
#include <cstdlib>
#include <set>


using namespace std;


OpenStackEntry::OpenStackEntry(state_var_t *_state_buffer):
        last_expanded_operator(-1),
        state_buffer(_state_buffer) {
    g_successor_generator->generate_applicable_ops(State(_state_buffer),
                                                   applicable_operators);
}

const Operator *OpenStackEntry::next_applicable_operator() {
    if (last_expanded_operator + 1 < applicable_operators.size()) {
        return applicable_operators[++last_expanded_operator];
    }
    return NULL;
}

void OpenStackEntry::reset() {
    last_expanded_operator = -1;
}


IDAStarSearch::IDAStarSearch(const Options &opts)
        : SearchEngine(opts),
        remember_pruned_nodes(opts.get<bool>("remember_pruned_nodes")),
        heuristic(opts.get<Heuristic *>("heuristic")) {
}

void IDAStarSearch::initialize() {
    cout << "Conducting IDA* search with bound = " << bound << endl;

    assert(heuristic);
    search_progress.add_heuristic(heuristic);

    SearchNode node = search_space.get_node(*g_initial_state);
    heuristic->evaluate(node.get_state());
    search_progress.inc_evaluated_states();
    search_progress.inc_evaluations(1);
    if (heuristic->is_dead_end()) {
        cout << "Initial state is a dead end." << endl;
    } else {
        search_progress.get_initial_h_values();
        search_progress.check_h_progress(0);
        layer = 0;
        node.open_initial(heuristic->get_value());
        if (remember_pruned_nodes) {
            next_layer_open_stack.push_back(OpenStackEntry(node.get_state_buffer()));
        }
        current_bound = heuristic->get_value();
        restart_with_next_bound(heuristic->get_value());
    }
}

int IDAStarSearch::restart_with_next_bound(int _current_bound) {
    search_progress.report_f_value(current_bound);
    if (_current_bound > bound || _current_bound == numeric_limits<int>::max()) {
        return FAILED;
    }
    layer++;
    current_bound = _current_bound;
    cout << "Next bound for IDA* search is " << current_bound << endl;
    assert(open_stack.empty());
    if (remember_pruned_nodes) {
        next_layer_open_stack.swap(open_stack);
        next_layer_open_stack.clear();
        for (int i=0; i<open_stack.size(); ++i) {
            open_stack[i].reset();
            SearchNode node = search_space.get_node(State(open_stack[i].state_buffer));
            if (node.is_closed()) {
                node.reopen();
            }
        }
    } else {
        SearchNode node = search_space.get_node(*g_initial_state);
        if (node.is_closed()) {
            node.reopen();
        }
        open_stack.push_back(OpenStackEntry(node.get_state_buffer()));
    }
    assert(!open_stack.empty());
    return IN_PROGRESS;
}

void IDAStarSearch::statistics() const {
    search_progress.print_statistics();
    search_space.statistics();
}


int IDAStarSearch::step() {
    if (current_bound == numeric_limits<int>::max()) {
        return FAILED;
    }
    int next_bound = numeric_limits<int>::max();
    while (true) {
        if (open_stack.empty()) {
            return restart_with_next_bound(next_bound);
        }
        OpenStackEntry *n = &open_stack.back();
        const Operator *op = n->next_applicable_operator();
        if (!op) {
            // This node was fully explored, backtrack by walking up the stack
            backtrack();
            continue;
        }

        State s(n->state_buffer);
        SearchNode node = search_space.get_node(s);
        int node_f = node.get_real_g() + node.get_h();
        if (node_f > current_bound) {
            // This node does not look promising enough, backtrack by walking up the stack
            // This can happen if the node was pruned in the last iteration
            next_bound = min(next_bound, node_f);
            if (remember_pruned_nodes) {
                next_layer_open_stack.push_back(*n);
            }
            backtrack();
            continue;
        }
        State succ_state(s, *op);
        SearchNode succ_node = search_space.get_node(succ_state);
        search_progress.inc_generated();

        // Previously encountered dead end. Don't re-evaluate.
        if (succ_node.is_dead_end())
            continue;
        int succ_g = node.get_real_g() + op->get_cost();
        bool check_goal = false;
        if (succ_node.is_new()) {
            // We have not seen this state before.
            // Evaluate and create a new node.
            heuristic->reach_state(s, *op, succ_node.get_state());
            heuristic->evaluate(succ_node.get_state());
            search_progress.inc_evaluated_states();
            search_progress.inc_evaluations(1);
            if (heuristic->is_dead_end()) {
                succ_node.mark_as_dead_end();
                search_progress.inc_dead_ends();
                continue;
            }
            // always open the node (even if it will be pruned) to avoid having
            // to recompute the heuristic if it is encountered again
            int succ_h = heuristic->get_heuristic();
            succ_node.open(succ_h, node, op);
            succ_node.set_layer(layer);
            check_goal = true;
            search_progress.check_h_progress(succ_g);
            open_stack.push_back(OpenStackEntry(succ_node.get_state_buffer()));
        } else if (succ_node.get_real_g() > succ_g ||
                   (succ_node.get_real_g() == succ_g && succ_node.get_layer() < layer)) {
            // We found a new cheapest path to an open or closed state.
            // cheaper paths to open states mean negative or zero cost cycles and are ignored.
            // TODO maybe handle this better
            if (succ_node.is_closed()) {
                check_goal = (succ_node.get_real_g() > succ_g);
                succ_node.set_layer(layer);
                search_progress.inc_reopened();
                succ_node.reopen(node, op);
                open_stack.push_back(OpenStackEntry(succ_node.get_state_buffer()));
            }
        }
        if (check_goal && check_goal_and_set_plan(succ_node.get_state())) {
            return SOLVED;
        }
    }
}


void IDAStarSearch::backtrack() {
    OpenStackEntry &n = open_stack.back();
    search_progress.inc_expanded();
    SearchNode node = search_space.get_node(State(n.state_buffer));
    node.close();
    heuristic->finished_state(node.get_state(), node.get_g() + node.get_h(), false);
    open_stack.pop_back();
}







static SearchEngine *_parse(OptionParser &parser) {
    parser.add_option<Heuristic *>
        ("heuristic", "Heuristic evaluator for node pruning");
    parser.add_option<bool>("remember_pruned_nodes", false,
                            "Keep a list of pruned nodes. This makes the IDA* search behave more like an A* search.");
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    IDAStarSearch *engine = 0;
    if (!parser.dry_run()) {
        engine = new IDAStarSearch(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin("ida_star", _parse);
