function cost=trainNN(numClasses, learningRate, beta1, beta2, imgDepth, filterSize, numFilter1, numFilter2, batchSize, numEpoch)
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Data preprocessing and normalisation
%
load Dataset.mat;

X=HOGValuesTRAIN;

% Normalize and standardize over N

X=permute(X,[4,3,2,1]);
X2=reshape(X,size(X,1),[]);
X=(X2' - floor(mean(X2,'omitnan'))')';
% X=(X / floor(std(X,1,1,'omitnan')));
B=floor(std(X,1,1,'omitnan'));

% Start of standard deviation normalisation
ispos = B>0;
C = X;
C(:,ispos) = X(:,ispos) ./ B(ispos);
X=C;
clear C B ispos;

X=reshape(X,size(X,1),imgDepth,20,20);

y=yTrain;
y=ones(length(y),1,20,20).*y;
dataTrain=horzcat(X,y);

%%%%%%%%%%%%%%%%%
dataTrain=dataTrain(randperm(size(dataTrain,1)),:,:,:);

f1=[numFilter1,imgDepth,filterSize,filterSize];
f2=[numFilter2,numFilter2,filterSize,filterSize];

w3=[100,288];
w4=[95,100];

f1=initFilter(f1);
f2=initFilter(f2);
w3=initWeights(w3);
w4=initWeights(w4);
 
b1=zeros(size(f1,1),1);
b2=zeros(size(f2,1),1);
b3=zeros(size(w3,1),1);
b4=zeros(size(w4,1),1);

params={f1,f2,w3,w4,b1,b2,b3,b4};

cost=0;

learningRate
batchSize
cnt=1;

for i=1:numEpoch
    dataTrain=dataTrain(randperm(size(dataTrain,1)),:,:,:);
    for j=1:batchSize:size(dataTrain,1)-batchSize
        batches{cnt}=dataTrain([j:j + batchSize-1],:,:,:);
        cnt=cnt+1;
    end
    length(batches)
    toc
    for j=1:length(batches)
    [params,cost]=adamGD(batches{j},numClasses,learningRate,beta1,beta2,params,cost);
    disp(cost(end));
    end
end
save('param.mat','params','cost');

toc
end
