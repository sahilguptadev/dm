go:-
write("enter 1st num -> "),read(X),nl,
write("enter 2st num -> "),read(Y),nl,
max(X,Y).

max(X,Y):-
X>=Y -> R is X,write(R),write('  '),write("is greater than "),write(Y);
R is Y,write(R),write(' '),write("is greater than "),write(X).