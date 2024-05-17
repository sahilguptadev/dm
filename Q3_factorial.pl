# factorial
go:-
write("Enter Number -> "),read(N),write("Factorial of "),
write(N),write(" is -> "),factorial(N,ANS),write(ANS).

factorial(0,1).
factorial(N,ANS):-
M is N-1,
factorial(M,K),
ANS is N*K.