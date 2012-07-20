#include "DepthEvaluator.h"
#include "option_parser.h"
#include "plugin.h"


static ScalarEvaluator *_parse(OptionParser &parser) {
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new DepthEvaluator(opts);
}


static Plugin<ScalarEvaluator> _plugin("depth", _parse);

static ScalarEvaluator *_parseIDEvaluator(OptionParser &parser) {
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new IDEvaluator(opts);
}


static Plugin<ScalarEvaluator> _plugin_("id", _parseIDEvaluator);
