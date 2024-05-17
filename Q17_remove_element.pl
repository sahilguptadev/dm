removen([H|List],1,Y,List):- write("Element at "),write(Y),write("th position --> "),write(H),nl,nl.
removen([H|List],Pos,Y,[H|Result]):-
    Pos1 is Pos-1,
    removen(List,Pos1,Y,Result).

go:-
    write("Enter list -> "),read(L),nl,
    write("List -> "),write(L),nl,
    write("Enter position of element to be deleted ->  "),read(Y),nl,
    write("Position of element -> "),write(Y),nl,
    removen(L,Y,Y,R),nl,
    write("List after deleting "),write(Y),write("th element --> "),
    write(R).

