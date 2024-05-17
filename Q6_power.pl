#6 Write a Prolog program to implement power (Num,Pow, Ans) : where Num is raised to the power Pow to get Ans.

power(0,P,0):- P>0.
power(X,0,1):- X>0.
power(X,P,A):- X>0,P>0,P1 is P-1,
power(X,P1,Ans),
A is Ans*X.

go:-
write("Enter X -> "),read(X),nl,
write("Enter power of X-> "),read(Y),nl,
write(X),write(" ^ "),write(Y),write(" is ")
,power(X,Y,ANS),write(ANS).