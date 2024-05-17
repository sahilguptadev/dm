go:-
write("1st num -> "),read(X),nl,
write("2nd num -> "),read(Y),nl,
add(X,Y).


add(X,Y):-
S is X+Y,
write("Sum -> "),write(S),nl.
