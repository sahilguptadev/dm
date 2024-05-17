insert_nth(I, 1, L, [I|L]).
insert_nth(I, N, [H|T], [H|R]) :-
    N > 1, 
    N1 is N - 1, 
    insert_nth(I, N1, T, R).

go:-
    write('Enter the list (in square brackets): '),read(L),nl,
    write('Enter the item to insert: '),read(I),nl,
    write('Enter the position to insert the element: '),read(N),nl,
    write(' List --> '),write(L),nl,
    write(' Element to Insert --> '),write(I),nl,
    write(' Position of element --> '),write(N),nl,
    insert_nth(I, N, L, R),
    write('The resulting list is: '),
    write(R), nl.
