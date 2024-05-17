
gcd(X,0,X).
gcd(X,Y,Z):- 
 R is mod(X,Y),
 gcd(Y,R,Z).

go:-
write("Enter 1st number -> "),read(X),nl,
write("Enter 2nd number -> "),read(Y),nl,
write("GCD of "),write(X),write(" and ")
,write(Y),write(" is "),
gcd(X,Y,ANS),write(ANS).