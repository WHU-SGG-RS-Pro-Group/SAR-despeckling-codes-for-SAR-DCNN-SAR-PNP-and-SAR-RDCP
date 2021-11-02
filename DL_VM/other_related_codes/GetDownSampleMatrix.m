function D = GetDownSampleMatrix(m,n,nSize);

M = m*nSize;
N = n*nSize;
MN = M*N;
mn = m*n;
A = sparse(m,M); 
B = sparse(n,N);
%C = sparse(MN,mn);
D = sparse(MN,mn);
A=kron(eye(m),ones(1,nSize))';
B=kron(eye(n),ones(1,nSize))';
D=kron(sparse(A),sparse(B))/(nSize*nSize);

% for i = 1:m
%     for j = 1:n
%         X = nSize*(i-1);
%         Y = nSize*(j-1);
%         Block_ID = (i-1)*n+j;
%         for p = 1:nSize
%             for q = 1:nSize
%                 X_buf = X+p;
%                 Y_buf = Y+q;
%                 Pos = N*(X_buf-1) + Y_buf;
%                 D(Pos,Block_ID) = 1/(nSize*nSize);
%             end;
%         end;
%     end;
% end;

        