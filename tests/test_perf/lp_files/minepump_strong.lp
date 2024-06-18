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

%assumption -- initial_assumption
%	highwater=false&methane=false;

assumption(initial_assumption).

antecedent_holds(initial_assumption,0,S):-
	trace(S),
	timepoint(0,S).

consequent_holds(initial_assumption,0,S):-
	trace(S),
	timepoint(0,S),
	not_holds_at(highwater,0,S),
	not_holds_at(methane,0,S).

%assumption -- assumption1_1
%	G(PREV(pump=true)&pump=true->next(highwater=false));

assumption(assumption1_1).

antecedent_holds(assumption1_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at_prev(pump,T1,S),
	holds_at(pump,T1,S).

consequent_holds(assumption1_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	not_holds_at_weak_next(highwater,T1,S).

%assumption -- assumption2_1
%	G(highwater=false|methane=false);

assumption(assumption2_1).

antecedent_holds(assumption2_1,T1,S):-
	trace(S),
	timepoint(T1,S).

consequent_holds(assumption2_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	not_holds_at(highwater,T1,S).

consequent_holds(assumption2_1,T1,S):-
	trace(S),
	timepoint(T1,S),
	not_holds_at(methane,T1,S).

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

%---*** Signature  ***---

atom(highwater).
atom(methane).
atom(pump).


%---*** Violation Trace ***---

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


#show violation_holds/3.

#show assumption/1.

#show guarantee/1.

#show entailed/1.
