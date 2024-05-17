#8 Write a Prolog program to implement memb(X, L): to check whether X is a member of L or not.

memb(X,[X|_]).
memb(X,[_|T]):-memb(X,T).


go:-
write("Enter list [] -> "),read(X),nl,
write("Element to be searched -> "),read(Y),nl,
write("is "),write(Y),write(" present in "),
write(X),write(" --> "),memb(Y,X).
