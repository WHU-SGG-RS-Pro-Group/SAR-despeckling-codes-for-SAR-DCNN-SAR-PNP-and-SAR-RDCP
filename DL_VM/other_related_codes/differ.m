function Dd=differ(m)
A=zeros(m);
A(1,1)=-1;
A(1,2)=1;
A(m,m)=-1;
A(m,m-1)=1;
for i=2:m-1
    A(i,i-1)=1;
    A(i,i+1)=1;
    A(i,i)=-2;
end
Dd=A;

