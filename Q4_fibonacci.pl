# 4. Write a program in PROLOG to implement generate_fib(N,T) where T represents the Nth term of the fibonacci series.

generate_fib(0,1).
generate_fib(1,1).
generate_fib(N,T):- 
N1 is N-1, 
generate_fib(N1,T1),
N2 is N-2, 
generate_fib(N2,T2), 
T is T1+T2.


go:-
write("Enter nth number -> "),read(X),nl,write(X),
write("th Fibonaci number is -> "),
generate_fib(X,ANS),write(ANS).