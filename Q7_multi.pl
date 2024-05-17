#7 Prolog program to implement multi (N1, N2, R) : where N1 and N2 denotes the numbers to be multiplied and R represents the result.

multi(X,Y,R):-
R is X*Y.


go:-
write("Enter 1st number -> "),read(X),nl,
write("Enter 2nd number -> "),read(Y),nl,
write("Multiplication of "),write(X),
write(" and "),write(Y),write(" is "),
multi(X,Y,ANS),write(ANS).