# Write a Prolog program to implement reverse (L, R) where List L is original and List R is reversed list.

reverse([], Y, R) :-
    R = Y.
reverse([H|T] , Y, R) :-
    reverse(T, [H|Y], R).


go:-
write("Enter list [] -> "),read(X),nl,
write("reversed list "),write(X),write(" --> "),reverse(X,[],R),write(R).