function [gradients, loss]=ConvolutionMain(image, label, params, sConv, fPool, sPool)

filter1 = params{1};
filter2 = params{2};
weight3 = params{3};
weight4 = params{4};
bias1 = params{5};
bias2 = params{6};
bias3 = params{7};
bias4 = params{8};


%
% Forward operation
%

conv1=conv(image,filter1,bias1,sConv);
conv1(conv1<=0)=0; % ReLU filter

conv2=conv(conv1,filter2,bias2,sConv);
conv2(conv2<=0)=0;  % ReLU filter

pooled=maxpool(conv2,fPool,sPool);

[numberFilters2,yDimPool,xDimPool]=size(pooled);
newPooled=reshape(pooled,[xDimPool*yDimPool*numberFilters2,1]); %Flattened vector

l=weight3*newPooled + bias3; % dot product
l(l<=0)=0; % ReLU filter

output=weight4*l+bias4; % dot product


classProb=softMax(output);

%
% Loss
%

loss=crossEntopy(classProb,label);

%
% Backward Operation
%

delOut= classProb - label;

delWeight4=delOut*l';

delBias4=reshape(sum(delOut,2), size(bias4));


del_l=weight4'*delOut;

del_l(del_l<=0) = 0; %BackPropagate through ReLU

delWeight3=del_l*newPooled';

delBias3=reshape(sum(del_l,2),size(bias3));


delNewPooled=weight3'*del_l;

delPool=reshape(delNewPooled,size(pooled));


delConv2=backwardMaxpool(delPool,conv2,fPool,sPool);

delConv2(delConv2<=0)=0;

[delConv1,delFilter2,delBias2]= backwardConvolution(delConv2,conv1,filter2,sConv);

delConv1(delConv1<=0)=0;

[delImage, delFilter1, delBias1]=backwardConvolution(delConv1, image, filter1, sConv);

gradients={delFilter1, delFilter2, delWeight3, delWeight4, delBias1, delBias2, delBias3, delBias4};

end

