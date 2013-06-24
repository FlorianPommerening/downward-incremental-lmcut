#include "fringe_search.h"
#include "globals.h"
#include "plugin.h"
#include "successor_generator.h"

using namespace std;

FringeSearch::FringeSearch(const Options &opts)
    : SearchEngine(opts),
    heuristic(opts.get<Heuristic *>("heuristic")) {
}

void FringeSearch::initialize() {
    cout << "Conducting fringe search" << endl;
    assert(heuristic != NULL);
    search_progress.add_heuristic(heuristic);
    heuristic->evaluate(*g_initial_state);
    search_progress.inc_evaluated_states();
    search_progress.inc_evaluations();
    if (heuristic->is_dead_end()) {
        cout << "Initial state is a dead end." << endl;
    } else {
        search_progress.get_initial_h_values();
        SearchNode node = search_space.get_node(*g_initial_state);
        int initial_h = heuristic->get_heuristic();
        node.open_initial(initial_h);
        search_progress.check_h_progress(0);
        f_limit = initial_h;
        fringe.push_back(node.get_state_buffer());
        node.set_fringe_it(fringe.begin());
    }
}

int FringeSearch::step() {
    if (fringe.empty()) {
        return FAILED;
    }
    search_progress.report_f_value(f_limit);
    int f_min = numeric_limits<int>::max();
    Fringe::iterator it = fringe.begin();
    while (it != fringe.end()) {
        State s = State(*it);
        SearchNode node = search_space.get_node(s);
        node.close();
        search_progress.inc_expanded();
        int f = node.get_g() + node.get_h();
        if (f > f_limit) {
            f_min = min(f, f_min);
            ++it;
            continue;
        }
        if (check_goal_and_set_plan(s)) {
            return SOLVED;
        }
        vector<const Operator *> applicable_ops;
        g_successor_generator->generate_applicable_ops(s, applicable_ops);
        for (int i = applicable_ops.size() - 1; i >= 0; i--) {
            const Operator *op = applicable_ops[i];
            if ((node.get_real_g() + op->get_cost()) >= bound)
                continue;
            State succ_state(s, *op);
            search_progress.inc_generated();
            SearchNode succ_node = search_space.get_node(succ_state);
            if (succ_node.is_dead_end())
                continue;
            int g_succ = node.get_g() + get_adjusted_cost(*op);
            if (!succ_node.is_new() && g_succ >= succ_node.get_g()) {
                continue;
            }
            if (succ_node.is_new()) {
                heuristic->reach_state(s, *op, succ_node.get_state());
                heuristic->evaluate(succ_node.get_state());
                search_progress.inc_evaluated_states();
                search_progress.inc_evaluations();
                if (heuristic->is_dead_end()) {
                    succ_node.mark_as_dead_end();
                    search_progress.inc_dead_ends();
                    continue;
                }
                int succ_h = heuristic->get_heuristic();
                succ_node.open(succ_h, node, op);
                search_progress.check_h_progress(succ_node.get_g());
            } else if (succ_node.is_open()) {
                assert(!(s == succ_state));
                fringe.erase(succ_node.get_fringe_it());
                succ_node.update_parent(node, op);
            } else if (succ_node.is_closed()) {
                succ_node.reopen(node, op);
                search_progress.inc_reopened();
            }
            // list::insert adds elements before the given iterator, so we copy and
            // increment the current iterator.
            Fringe::iterator copy = it;
            ++copy;
            Fringe::iterator succ_it = fringe.insert(copy, succ_node.get_state_buffer());
            succ_node.set_fringe_it(succ_it);
        }
        Fringe::iterator next_it = it;
        ++next_it;
        fringe.erase(it);
        it = next_it;
        // HACK try to maintain only met information for boundary nodes
        // only useful for incremental lmcut atm
        heuristic->finished_state(s, node.get_real_g() + node.get_h(), false);
    }
    f_limit = f_min;
    return IN_PROGRESS;
}

void FringeSearch::statistics() const {
    search_progress.print_statistics();
    search_space.statistics();
}

static SearchEngine *_parse(OptionParser &parser) {
    parser.add_option<Heuristic *>("heuristic");
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    FringeSearch *engine = 0;
    if (!parser.dry_run()) {
        engine = new FringeSearch(opts);
    }

    return engine;
}

static Plugin<SearchEngine> _plugin("fringe", _parse);
