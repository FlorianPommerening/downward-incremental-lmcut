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
        open_stack.push_back(OpenStackEntry(node.get_state_buffer()));
        current_bound = heuristic->get_value();
        restart_with_next_bound(heuristic->get_value());
    }
}

int IDAStarSearch::restart_with_next_bound(int _current_bound) {
    // HACK
    if (_current_bound != 1) {
        search_progress.report_f_value(current_bound);
    }
    if (_current_bound > bound || _current_bound == numeric_limits<int>::max()) {
        return FAILED;
    }
    layer++;
    current_bound = _current_bound;
    cout << "Next bound for IDA* search is " << current_bound << endl;
    assert(open_stack.size() == 1);
    open_stack[0].reset();
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
        OpenStackEntry *n = &open_stack.back();
        const Operator *op = n->next_applicable_operator();
        if (!op) {
            // this node was fully explored, backtrack by walking up the stack
            while (true) {
                if (open_stack.size() == 1) {
                    // never close the initial node (it is reused for the next iteration)
                    return restart_with_next_bound(next_bound);
                }
                else {
                    backtrack();
                }
                n = &open_stack.back();
                op = n->next_applicable_operator();
                // this node was not fully explored yet, stop backtracking
                if (op){
                    break;
                }
            }
        }

        State s(n->state_buffer);
        SearchNode node = search_space.get_node(s);
        State succ_state(State(n->state_buffer), *op);
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
            if (heuristic->is_dead_end()) {
                succ_node.mark_as_dead_end();
                search_progress.inc_dead_ends();
                continue;
            }
            search_progress.inc_evaluated_states();
            search_progress.inc_evaluations(1);
            // always open the node (even if it will be pruned) to avaoid having
            // to recompute the heuristic if it is encountered again
            int succ_h = heuristic->get_heuristic();
            succ_node.open(succ_h, node, op);
            succ_node.set_layer(layer);

            int succ_f = succ_g + succ_h;
            if (succ_f > current_bound) {
                next_bound = min(next_bound, succ_f);
                succ_node.close();
                heuristic->finished_state(succ_node.get_state(), succ_f, true);
                continue;
            }
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
                int succ_f = succ_g + succ_node.get_h();
                if (succ_f > current_bound) {
                    next_bound = min(next_bound, succ_f);
                    heuristic->finished_state(succ_node.get_state(), succ_f, true);
                    continue;
                }
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
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    IDAStarSearch *engine = 0;
    if (!parser.dry_run()) {
        engine = new IDAStarSearch(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin("ida_star", _parse);
