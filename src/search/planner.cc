#include "globals.h"
#include "operator.h"
#include "option_parser.h"
#include "ext/tree_util.hh"
#include "timer.h"
#include "utilities.h"
#include "search_engine.h"

// HACK
#include "memory_tracking_allocator.h"


#include <iostream>
using namespace std;



int main(int argc, const char **argv) {
    register_event_handlers();

    if (argc < 2) {
        cout << OptionParser::usage(argv[0]) << endl;
        exit(1);
    }

    if (string(argv[1]).compare("--help") != 0)
        read_everything(cin);

    SearchEngine *engine = 0;

    //the input will be parsed twice:
    //once in dry-run mode, to check for simple input errors,
    //then in normal mode
    try {
        OptionParser::parse_cmd_line(argc, argv, true);
        engine = OptionParser::parse_cmd_line(argc, argv, false);
    } catch (ParseError &pe) {
        cout << pe << endl;
        exit(1);
    }

    Timer search_timer;
    engine->search();
    search_timer.stop();
    g_timer.stop();

    engine->save_plan_if_necessary();
    engine->statistics();
    engine->heuristic_statistics();

    //HACK
    cout << "Tracked memory: " << g_memory_tracking_allocated / 1024 << " KB" << endl;
    cout << "Tracked memory (states 1): " << g_memory_tracking_states_1 / 1024 << " KB" << endl;
    cout << "Tracked memory (states 2): " << g_memory_tracking_states_2 / 1024 << " KB" << endl;
    cout << "Tracked memory (states ht): " << g_memory_tracking_states_ht / 1024 << " KB" << endl;
    cout << "Tracked memory (landmarks bit array): " << g_memory_tracking_landmarks_bit_array / 1024 << " KB" << endl;
    cout << "Tracked memory (landmarks indices): " << g_memory_tracking_landmarks_indices / 1024 << " KB" << endl;

    cout << "Search time: " << search_timer << endl;
    cout << "Total time: " << g_timer << endl;

    return engine->found_solution() ? 0 : 1;
}
