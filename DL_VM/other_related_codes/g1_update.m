function GG_h=g1_update(m,n)
G_1=speye(m*(n-1));
G_2=sparse(m*(n-1),m);
G_3=speye(m);
G_uint1=[-G_1,G_2]+[G_2,G_1];
G_uint2=[G_3,G_2']+[G_2',-G_3];
GG_h=[G_uint1;G_uint2];
