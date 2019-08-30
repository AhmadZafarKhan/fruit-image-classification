function [pred, prob]=predict(data, filter1,filter2,weight3,weight4,bias1,bias2,bias3,bias4);

sConv=1;
fPool=2;
sPool=2;


conv1=conv(data,filter1,bias1,sConv);
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

prob=max(classProb,[],'omitnan');
pred=find(classProb==prob,1);

end