merge(X,[],X).
merge([],[],[]).
merge([],Y,Y).

merge([Hx|X],[Hy|Y],[Hx|Z]):- Hx<Hy,!,merge(X,[Hy|Y],Z).
merge([Hx|X],[Hy|Y],[Hx|Z]):- Hx==Hy,!,merge(X,Y,Z).
merge([Hx|X],[Hy|Y],[Hy|Z]):- Hx>Hy,!,merge([Hx|X],Y,Z).

go:-
    write("Enter Ordered list L1 -> "),read(L1),nl,
    write("Ordered List L1-> "),write(L1),nl,
    write("Enter Ordered list L2 -> "),read(L2),nl,
    write("Ordered List L2-> "),write(L2),nl,
    merge(L1,L2,R),
    write("Merged Ordered List --> "),write(R).

