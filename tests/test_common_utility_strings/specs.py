spec_perf = """
module Minepump

env boolean highwater;
env boolean methane;
sys boolean pump;

assumption -- initial_assumption
    !highwater & !methane;

guarantee -- initial_guarantee
    !pump;

guarantee -- guarantee1_1
	G(highwater&!methane->next(pump));

guarantee -- guarantee2_1
	G(methane->next(!pump));

assumption -- assumption1_1
	G(PREV(pump)&pump->next(!highwater));
    """

spec_fixed_perf = """
module Minepump

env boolean highwater;
env boolean methane;
sys boolean pump;

assumption -- initial_assumption
    highwater=false & methane=false;

guarantee -- initial_guarantee
    pump=false;

guarantee -- guarantee1_1
         G(highwater=true->methane=true|next(pump=true));


guarantee -- guarantee2_1
    G(methane=true->next(pump=false));

assumption -- assumption1_1
    G(PREV(pump=true)&pump=true->next(highwater=false));

assumption -- assumption2_1
    G(highwater=false-> highwater=false|methane=false);
    """
spec_fixed_imperf = """
module Minepump

env boolean highwater;
env boolean methane;
sys boolean pump;

assumption -- initial_assumption
    highwater=false & methane=false;

guarantee -- initial_guarantee
    pump=false;

guarantee -- guarantee1_1
		G(highwater=true->highwater=true|next(pump=true));


guarantee -- guarantee2_1
	G(methane=true->next(pump=false));

assumption -- assumption1_1
	G(PREV(pump=true)&pump=true->next(highwater=false));

assumption -- assumption2_1
	G(highwater=false-> highwater=false|methane=false);
"""
spec_strong = """
module Minepump

env boolean highwater;
env boolean methane;
sys boolean pump;

assumption -- initial_assumption
    !highwater & !methane;

guarantee -- initial_guarantee
    !pump;

guarantee -- guarantee1_1
	G(highwater->next(pump));

guarantee -- guarantee2_1
	G(methane->next(!pump));

assumption -- assumption1_1
	G(PREV(pump)&pump->next(!highwater));

assumption -- assumption2_1
	G(!highwater|!methane);
"""

spec_strong_asm_w = """
module Minepump

env boolean highwater;
env boolean methane;
sys boolean pump;

assumption -- initial_assumption
    !highwater & !methane;

guarantee -- initial_guarantee
    !pump;

guarantee -- guarantee1_1
	G(highwater->next(pump));

guarantee -- guarantee2_1
	G(methane->next(!pump));

assumption -- assumption1_1
	G(PREV(pump)&pump->next(!highwater));

assumption -- assumption2_1
    G(highwater=false-> highwater=false|methane=false);
"""

spec_asm_stronger_gar_eq = """
module Minepump

env boolean highwater;
env boolean methane;
sys boolean pump;

assumption -- initial_assumption
    highwater=false & methane=false;

guarantee -- initial_guarantee
    pump=false;

guarantee -- guarantee1_1 -- modified to correct
		G(highwater=true->methane=true|next(pump=true));


guarantee -- guarantee2_1
	G(methane=true->next(pump=false));

assumption -- assumption1_1
	G(PREV(pump=true)&pump=true->next(highwater=false));

assumption -- assumption2_1 -- modified
	G(pump=true-> highwater=false|methane=false);
"""

spec_asm_eq_gar_weaker = """
module Minepump

env boolean highwater;
env boolean methane;
sys boolean pump;

assumption -- initial_assumption
    highwater=false & methane=false;

guarantee -- initial_guarantee
    pump=false;

guarantee -- guarantee1_1 -- trivialised
		G(highwater=true->highwater=true|next(pump=true));


guarantee -- guarantee2_1
	G(methane=true->next(pump=false));

assumption -- assumption1_1
	G(PREV(pump=true)&pump=true->next(highwater=false));

assumption -- assumption2_1 -- trivialised
	G(highwater=false-> highwater=false|methane=false);
"""
