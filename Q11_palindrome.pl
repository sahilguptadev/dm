# Write a program in PROLOG to implement palindrome (L) which checks whether a list L is a palindrome or not.

palind([]):- write('palindrome').
palind([_]):- write('palindrome').
palind(L) :-append([H|T], [H], L),palind(T);
write('Not a palindrome').

go:-
write("Enter list [] -> "),read(X),nl,
write("List = "),write(X),nl,
write("Plaindrome or not -> "),palind(X).