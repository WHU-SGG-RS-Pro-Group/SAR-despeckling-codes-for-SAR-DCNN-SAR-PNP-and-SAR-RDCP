function D=laps(m,n)
Da=sparse(differ(n));
Ib=speye(m);
D1=kron(Da,Ib);
Db=sparse(differ(m));
Ia=speye(n);
D2=kron(Ia,Db);
D=D1+D2;
