evenlength:-
 write('given list is in even length').
oddlength:-
 write('given list is in odd length').
oddeven([_|T]):-
 length(T,L),
 L>=0 ->
 (
  L1 is L+1,
  L2 is mod(L1,2),
  L2=:=0 ->
   evenlength;
   oddlength
 ).

go:-

write("Enter list [] -> "),read(X),nl,
write("List = "),write(X),nl,
write("Odd or Even Length -> "),oddeven(X).
