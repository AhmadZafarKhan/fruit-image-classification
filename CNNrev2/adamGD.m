function [params1,rCost]=adamGD(batch, numClasses, learningRate, beta1, beta2, params, rCost)
%
% Update parameters using Adams gradient descent
%

filter1 = params{1};
filter2 = params{2};
weight3 = params{3};
weight4 = params{4};
bias1 = params{5};
bias2 = params{6};
bias3 = params{7};
bias4 = params{8};


A=batch(:,[1:end-1],:,:);
Y=batch(:,end,1,1);

cost=0;

batchSize=size(A,1);

df1 = zeros(size(filter1));
df2 = zeros(size(filter2));
dw3 = zeros(size(weight3));
dw4 = zeros(size(weight4));
db1 = zeros(size(bias1));
db2 = zeros(size(bias2));
db3 = zeros(size(bias3));
db4 = zeros(size(bias4));

v1 = zeros(size(filter1));
v2 = zeros(size(filter2));
v3 = zeros(size(weight3));
v4 = zeros(size(weight4));
bv1 = zeros(size(bias1));
bv2 = zeros(size(bias2));
bv3 = zeros(size(bias3));
bv4 = zeros(size(bias4));

s1 = zeros(size(filter1));
s2 = zeros(size(filter2));
s3 = zeros(size(weight3));
s4 = zeros(size(weight4));
bs1 = zeros(size(bias1));
bs2 = zeros(size(bias2));
bs3 = zeros(size(bias3));
bs4 = zeros(size(bias4));

for i=1:batchSize
    x=A(i,:,:,:);
    x=reshape(x,[4,20,20]);
    y=eye(numClasses);
    y=y(Y(i),:);
    y=reshape(y,[numClasses,1]);
    
    [grads,loss]=ConvolutionMain(x,y,params,1,2,2);
    dF1=grads{1};
    dF2=grads{2};
    dW3=grads{3};
    dW4=grads{4};
    dB1=grads{5};
    dB2=grads{6};
    dB3=grads{7};
    dB4=grads{8};
    
    df1=df1+dF1;
    db1=db1+dB1;
    df2=df2+dF2;
    db2=db2+dB2;
    dw3=dw3+dW3;
    db3=db3+dB3;
    dw4=dw4+dW4;
    db4=db4+dB4;
    
    cost=cost+loss;
    
end
    
v1=(beta1.*v1) + (1-beta1).*df1./batchSize;
s1=(beta2.*s1) + (1-beta2).*((df1./batchSize).^2);
filter1=filter1-(learningRate*(v1./sqrt(s1+(1*(10^(-7))))));

bv1=(beta1.*bv1) + (1-beta1).*db1./batchSize;
bs1=(beta2.*bs1) + (1-beta2).*((db1./batchSize).^2);
bias1=bias1-(learningRate.*(bv1./sqrt(bs1+(1*(10^(-7)))))); 

v2=(beta1.*v2) + (1-beta1).*df2./batchSize;
s2=(beta2.*s2) + (1-beta2).*((df2./batchSize).^2);
filter2=filter2-(learningRate.*(v2./sqrt(s2+(1*(10^(-7))))));

bv2=(beta1.*bv2) + (1-beta1).*db2/batchSize;
bs2=(beta2.*bs2) + (1-beta2).*((db2/batchSize).^2);
bias2=bias2-(learningRate.*(bv2./sqrt(bs2+(1*(10^(-7))))));     

v3=(beta1.*v3) + (1-beta1).*dw3/batchSize;
s3=(beta2.*s3) + (1-beta2).*((dw3/batchSize).^2);
weight3=weight3-(learningRate.*(v3./sqrt(s3+(1*(10^(-7))))));

bv3=(beta1.*bv3) + (1-beta1).*db3/batchSize;
bs3=(beta2.*bs3) + (1-beta2).*((db3./batchSize).^2);
bias3=bias3-(learningRate.*(bv3./sqrt(bs3+(1*(10^(-7))))));    

v4=(beta1.*v4) + (1-beta1).*dw4/batchSize;
s4=(beta2.*s4) + (1-beta2).*((dw4./batchSize).^2);
weight4=weight4-(learningRate.*(v4./sqrt(s4+(1*(10^(-7))))));

bv4=(beta1.*bv4) + (1-beta1).*db4./batchSize;
bs4=(beta2.*bs4) + (1-beta2).*((db4./batchSize).^2);
bias4=bias4-(learningRate*(bv4./sqrt(bs4+(1*(10^(-7))))));    

cost= cost/batchSize;
rCost=[rCost cost];

params1={filter1, filter2, weight3, weight4, bias1, bias2, bias3, bias4};

end

