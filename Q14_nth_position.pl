nth_element(1,[H|_],H).
nth_element(N,[_|T],X):- N1 is N-1,nth_element(N1,T,X).

go:-
write('Enter a list: '),
read(L),
write('Enter a position: '),
read(N),
(  
     nth_element(N, L, X)
    ->  write('The element at position '),
     write(N), write(' is: '), writeln(X);
    writeln('The position is out of range.')
).