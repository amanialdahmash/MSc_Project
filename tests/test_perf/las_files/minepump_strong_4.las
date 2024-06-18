#ilasp_script

max_solutions = 10

ilasp.cdilp.initialise()
solve_result = ilasp.cdilp.solve()

c_egs = None
if solve_result is not None:
  c_egs = ilasp.find_all_counterexamples(solve_result)

conflict_analysis_strategy = {
  'positive-strategy': 'all-ufs',
  'negative-strategy': 'single-as',
  'brave-strategy':    'all-ufs',
  'cautious-strategy': 'single-as-pair'
}

solution_count = 0

while solution_count < max_solutions and solve_result is not None:
  if c_egs:
    ce = ilasp.get_example(c_egs[0]['id'])
    constraint = ilasp.cdilp.analyse_conflict(solve_result['hypothesis'], ce['id'], conflict_analysis_strategy)
  
    # An example with recorded penalty of 0 is in reality an example with an
    # infinite penalty, meaning that it must be covered. Constraint propagation is,
    # therefore, unnecessary.
    if not ce['penalty'] == 0:
      c_eg_ids = list(map(lambda x: x['id'], c_egs))
      prop_egs = []
      if ce['type'] == 'positive':
        prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['positive'], 'strategy': 'cdpi-implies-constraint'})
      elif ce['type'] == 'negative':
        prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['negative'], 'strategy': 'neg-constraint-implies-cdpi'})
      elif ce['type'] == 'brave-order':
        prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['brave-order'],    'strategy': 'cdoe-implies-constraint'})
      else:
        prop_egs = [ce['id']]
  
      ilasp.cdilp.add_coverage_constraint(constraint, prop_egs)
  
    else:
      ilasp.cdilp.add_coverage_constraint(constraint, [ce['id']])

  solve_result = ilasp.cdilp.solve()

  if solve_result is not None:
    c_egs = ilasp.find_all_counterexamples(solve_result)
    if not c_egs:
      solution_count+=1
      debug_print(f'Solution {solution_count} (score {solve_result["expected_score"]})')
      print(ilasp.hypothesis_to_string(solve_result['hypothesis']))
      new_constraint_body = map(lambda x: f'nge_HYP({x})', solve_result["hypothesis"])
      # if you want to rule allow non-subset-minimal solutions uncomment this line and comment the one below.
      # new_constraint = f':- {",".join(new_constraint_body)}, #count' + "{ H : nge_HYP(H) }" + f' = {len(solve_result["hypothesis"])}.\n'
      new_constraint = f':- {",".join(new_constraint_body)}.\n'
      ilasp.cdilp.add_to_meta_program(new_constraint)


if solution_count == 0:
  print('UNSATISFIABLE')

ilasp.stats.print_timings()

#end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mode Declaration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#modeh(consequent_exception(const(expression_v), var(time), var(trace))).
#modeb(2,holds_at(const(usable_atom), var(time), var(trace)), (positive)).
#modeb(2,not_holds_at(const(usable_atom), var(time), var(trace)), (positive)).
#modeb(2,holds_at_eventually(const(usable_atom), var(time), var(time), var(trace)), (positive)).
#modeb(2,not_holds_at_eventually(const(usable_atom), var(time), var(time), var(trace)), (positive)).
#constant(usable_atom,highwater).
#constant(usable_atom,methane).
#constant(usable_atom,pump).
#constant(expression_v, guarantee1_1).
#constant(expression_v, guarantee2_1).
#bias("
:- constraint.
:- head(consequent_exception(_,V1,V2)), body(holds_at(_,V3,V4)), (V3, V4) != (V1, V2).
:- head(consequent_exception(_,V1,V2)), body(not_holds_at(_,V3,V4)), (V3, V4) != (V1, V2).
:- head(consequent_exception(_,V1,V2)), body(holds_at_eventually(_,V3,_,V4)), (V3, V4) != (V1, V2).
:- head(consequent_exception(_,V1,V2)), body(not_holds_at_eventually(_,V3,_,V4)), (V3, V4) != (V1, V2).
:- head(consequent_exception(_,_,_)), body(holds_at_eventually(_,V1,V2,_)), V1 = V2.
:- head(consequent_exception(_,_,_)), body(not_holds_at_eventually(_,V1,V2,_)), V1 = V2.
:- body(holds_at_eventually(_,_,V1,_)), body(holds_at_eventually(_,_,V2,_)), V1 != V2.
:- head(consequent_exception(guarantee1_1,V1,V2)), body(holds_at_eventually(_,_,_,_)).
:- head(consequent_exception(guarantee1_1,V1,V2)), body(not_holds_at_eventually(_,_,_,_)).
:- head(consequent_exception(guarantee2_1,V1,V2)), body(holds_at_eventually(_,_,_,_)).
:- head(consequent_exception(guarantee2_1,V1,V2)), body(not_holds_at_eventually(_,_,_,_)).
:- head(consequent_exception(guarantee1_1,V1,V2)), body(holds_at_weak_next(pump,V1,V2)).
:- head(consequent_exception(guarantee2_1,V1,V2)), body(not_holds_at_weak_next(pump,V1,V2)).
").

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Background Knowledge
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---*** Domain independent Axioms ***---

next_timepoint_exists(T1,S):-
    trace(S),
    timepoint(T1,S),
    timepoint(T2,S),
    next(T2,T1,S).

holds_at_weak_next(F,P,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    not next_timepoint_exists(P,S).

not_holds_at_weak_next(F,P,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    not next_timepoint_exists(P,S).

holds_at_weak_next(F,P,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	next(P2,P,S),
	holds_at(F,P2,S).

not_holds_at_weak_next(F,P,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	next(P2,P,S),
	not_holds_at(F,P2,S).

holds_at_next(F,P,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	next(P2,P,S),
	holds_at(F,P2,S).

not_holds_at_next(F,P,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	next(P2,P,S),
	not_holds_at(F,P2,S).

holds_at_prev(F,P,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	next(P,P2,S),
	holds_at(F,P2,S).

not_holds_at_prev(F,P,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	next(P,P2,S),
	not_holds_at(F,P2,S).

after(T2,T1,S):-
    trace(S),
    timepoint(T1,S),
    timepoint(T2,S),
    next(T2,T1,S).

after(T3,T1,S):-
    trace(S),
    timepoint(T1,S),
    timepoint(T2,S),
    timepoint(T3,S),
    next(T2,T1,S),
    after(T3,T2,S).

holds_at_eventually(F,P,P2,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    after(P2,P,S),
    not next_timepoint_exists(P2,S).

not_holds_at_eventually(F,P,P2,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    after(P2,P,S),
    not next_timepoint_exists(P2,S).

holds_at_eventually(F,P,P,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    not next_timepoint_exists(P,S).

not_holds_at_eventually(F,P,P,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    not next_timepoint_exists(P,S).

holds_at_eventually(F,P,P2,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	timepoint(P2,S),
	after(P2,P,S),
	holds_at(F,P2,S).

not_holds_at_eventually(F,P,P2,S):-
	trace(S),
	atom(F),
	timepoint(P,S),
	timepoint(P2,S),
	after(P2,P,S),
	not_holds_at(F,P2,S).

eq(T1,T2):-
	trace(S),
	timepoint(T1,S),
	timepoint(T2,S),
	T1 == T2.

:- 	atom(F),
	trace(S),
	timepoint(T,S),
	not_holds_at(F,T,S),
	holds_at(F,T,S).

holds_non_vacuously(E, T1, S):-
	exp(E),
	trace(S),
	timepoint(T1,S),
	antecedent_holds(E, T1, S),
	consequent_holds(E, T1, S).

holds_vacuously(E, T1, S):-
	exp(E),
	trace(S),
	timepoint(T1,S),
	not antecedent_holds(E, T1, S).

holds(G, T, S):-
	timepoint(T,S),
	trace(S),
	exp(G),
	holds_non_vacuously(G, T, S).

holds(G, T, S):-
	timepoint(T,S), trace(S),
	exp(G),
	holds_vacuously(G, T, S).

violation_holds(G,T,S):-
	exp(G),
	trace(S),
	timepoint(T,S),
	not holds(G,T,S).
	%,
	%timepoint(T1,S),
	%next(T1,T,S).

violated(S):-
	exp(G),
	trace(S),
	timepoint(T,S),
	violation_holds(G,T,S).

entailed(S):-
	trace(S),
	not violated(S).

exp(E):-
	guarantee(E).

exp(E):-
	assumption(E).


% ---*** Domain dependent Axioms ***---

%guarantee -- initial_guarantee
%	pump=false;

guarantee(initial_guarantee).

antecedent_holds(initial_guarantee,0,S):-
	trace(S),
	timepoint(0,S).

consequent_holds(initial_guarantee,0,S):-
	trace(S),
	timepoint(0,S),
	not_holds_at(pump,0,S).

%guarantee -- guarantee1_1
%	G(highwater=true->next(pump=true));

guarantee(guarantee1_1).

antecedent_holds(guarantee1_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at(highwater,T1,S).

consequent_holds(guarantee1_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at_weak_next(pump,T1,S).

consequent_holds(guarantee1_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	consequent_exception(guarantee1_1,T1,S).

%guarantee -- guarantee2_1
%	G(methane=true->next(pump=false));

guarantee(guarantee2_1).

antecedent_holds(guarantee2_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at(methane,T1,S).

consequent_holds(guarantee2_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	not_holds_at_weak_next(pump,T1,S).

consequent_holds(guarantee2_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	consequent_exception(guarantee2_1,T1,S).

%---*** Signature  ***---

atom(highwater).
atom(methane).
atom(pump).


%---*** Violation Trace ***---

#pos({entailed(trace_name_0)},{},{
trace(trace_name_0).

timepoint(0,trace_name_0).
timepoint(1,trace_name_0).
next(1,0,trace_name_0).

not_holds_at(highwater,0,trace_name_0).
not_holds_at(methane,0,trace_name_0).
not_holds_at(pump,0,trace_name_0).
holds_at(highwater,1,trace_name_0).
holds_at(methane,1,trace_name_0).
not_holds_at(pump,1,trace_name_0).

}).

%---*** Violation Trace ***---

#pos({entailed(counter_strat_0)},{},{

% CS_Path: counter_strat_0

trace(counter_strat_0).

timepoint(0,counter_strat_0).
timepoint(1,counter_strat_0).
timepoint(2,counter_strat_0).
next(1,0,counter_strat_0).
next(2,1,counter_strat_0).

not_holds_at(highwater,0,counter_strat_0).
not_holds_at(methane,0,counter_strat_0).
not_holds_at(pump,0,counter_strat_0).
holds_at(highwater,1,counter_strat_0).
holds_at(methane,1,counter_strat_0).
holds_at(pump,1,counter_strat_0).
not_holds_at(highwater,2,counter_strat_0).
not_holds_at(methane,2,counter_strat_0).
not_holds_at(pump,2,counter_strat_0).

}).

