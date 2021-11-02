
%%
%network setting
clc
clear
useGPU      = 1;
totalIter=30;
format compact;
addpath(fullfile('data','utilities'));
showResult  = 1;
pauseTime   = 0;
modelName   = 'model_64_25_Res_Bnorm_Adam';

% for jj=51:60
epoch       =30;

load(fullfile('бнбн\data',modelName,[modelName,'-epoch-',num2str(epoch),'.mat']));
net1 = vl_simplenn_tidy(net);
net1.layers = net1.layers(1:end-1);
net1 = vl_simplenn_tidy(net1);
if useGPU
    net1 = vl_simplenn_move(net1, 'gpu') ;
end

%%


 


%%image path
filepaths1           =  [];
filepaths2           =  [];

folder1='data\QuickBird\MUL';
folder2='data\QuickBird\PAN';
filepaths1 = cat(1,filepaths1, dir(fullfile(folder1, '*.tif')));
filepaths2 = cat(1,filepaths2, dir(fullfile(folder2, '*.tif')));

%   
% for ii =1:length(filepaths2)
ii=4;
    reference=imread(fullfile(folder1,filepaths1(ii).name));
   pan=imread(fullfile(folder2,filepaths2(ii).name));
   reference=double(reference);
    pan=double(pan);
    pan=RSgenerate(pan,0,0);
    reference=RSgenerate(reference,0,0);
    [m,n,~]=size(reference);
 pan=imresize(pan,[m,n],'bicubic');
 clear m;
 clear n;
[m,~]=size(pan);
if mod(m,4)==0
    padding=0;
else
    padding=1;
end
pan=shave(pan,[padding,padding]);
reference=double(shave(reference,[padding,padding]));
pan=RSgenerate(pan,0,0);
mul=imresize(reference,1/4,'bicubic');
mul_up=imresize(mul,4,'bicubic');
[m,n,j]=size(mul_up);

P=GetDownSampleMatrix(m/4,n/4,4);
P=P';
[D,Dt] = defMDDt;
% for jj=1:100
%    for jjj=1:20
%%
output=imresize(mul,4,'bicubic');

lamda1=0.1;
lamda2=0.0001;
    
tol       =   0.0001;
[dz1,dz2]=D(pan);
dx1=zeros(size(output));
dx2=zeros(size(output));

%V1
for i=1:4
        [a,b]=D(output(:,:,i));
        dx1(:,:,i)=a;
        dx2(:,:,i)=b;
end
dV1=cat(3,dx1,dz1,dx2,dz2);
  dV1=im2single(dV1);
 if useGPU
dV1= gpuArray(dV1);
end
    res    = vl_simplenn(net1,dV1,[],[],'conserveMemory',true,'mode','test');
    im2= res(end).x;
    dV1=dV1+im2;
    if useGPU
       dV1 = gather(dV1);
    end
    dV1=double(dV1);
V1_h=dV1(:,:,1:4);
       V1_v=dV1(:,:,6:9);
       V1_h=reshape(V1_h,[m*n,j]);
       V1_v=reshape(V1_v,[m*n,j]);
       G1=g1_update(m,n);
       G2=g2_update(m,n);
       Q2=laps(m,n);
      
  
%%
%optimal solution
y=reshape(mul,[m*n/16,j]);
  A=P'*P+lamda1*(G1'*G1)+lamda1*(G2'*G2)+lamda2*(Q2'*Q2);
  Y=P'*y+lamda1*G1'*V1_h+lamda1*G2'*V1_v;
  for i=1:4
  X(:,i)=pcg(A,Y(:,i),[],50);
  end
  output=reshape(X,[m,n,j]);

reference2=uint8(255.*reference);
output2=uint8(255.*output);
mul_up2=uint8(255.*mul_up);

ergas=ERGAS(reference,output,4);
sam=spAngle(reference,output);
q=Q(reference,output);
psnr = PSNR(output2,reference2);
enviwrite(output,m,n,j,['бнбн/output_',num2str(ii)]);











