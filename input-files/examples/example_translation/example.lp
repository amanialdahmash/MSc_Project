%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Background Knowledge
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---*** Domain independent Axioms ***---

next_timepoint_exists(T1,S):-
    trace(S),
    timepoint(T1,S),
    timepoint(T2,S),
    next(T2,T1,S).

holds_at_next(F,P,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    not next_timepoint_exists(P,S).

not_holds_at_next(F,P,S):-
    trace(S),
    atom(F),
    timepoint(P,S),
    not next_timepoint_exists(P,S).

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

%assumption -- prev_and_next
%	G(PREV(a=true)->next(c=true));

assumption(prev_and_next).

antecedent_holds(prev_and_next,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at_prev(a,T1,S).

consequent_holds(prev_and_next,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at_next(c,T1,S).

%guarantee -- justice
%	G(a=true->b=true);

guarantee(justice).

antecedent_holds(justice,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at(a,T1,S).

consequent_holds(justice,T1,S):-
	trace(S),
	timepoint(T1,S),
	holds_at(b,T1,S).

%---*** Signature  ***---

atom(a).
atom(b).
atom(c).


%---*** Violation Trace ***---

trace(trace_name).

timepoint(0,trace_name).
timepoint(1,trace_name).
timepoint(2,trace_name).
next(1,0,trace_name).
next(2,1,trace_name).

    holds_at(a,0,trace_name).
    holds_at(b,0,trace_name).
not_holds_at(c,0,trace_name).
not_holds_at(a,1,trace_name).
not_holds_at(b,1,trace_name).
not_holds_at(c,1,trace_name).
not_holds_at(a,2,trace_name).
not_holds_at(b,2,trace_name).
not_holds_at(c,2,trace_name).


#show violation_holds/3.