maxlist([],0).

maxlist([Head|Tail],Max) :-
    maxlist(Tail,TailMax),
    Head > TailMax,
    Max is Head.

maxlist([Head|Tail],Max) :-
    maxlist(Tail,TailMax),
    Head =< TailMax,
    Max is TailMax.
go:-
write("Enter List L -> "),read(L),nl,
write("List -> "),write(L),nl,
write("Max Element in List is -> "),maxlist(L,M),write(M).