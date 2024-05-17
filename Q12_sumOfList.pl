
sumlist([],0).
  
sumlist([H|T],R):-
  sumlist(T,R1),
  R is H+R1.


go:-
write("Enter list [] -> "),
read(X),nl,
write("List = "),write(X),nl,
write("Sum of list -> "),
sumlist(X,Sum),write(Sum).