clear;
clc;

load param.mat

filter1 = params{1};
filter2 = params{2};
weight3 = params{3};
weight4 = params{4};
bias1 = params{5};
bias2 = params{6};
bias3 = params{7};
bias4 = params{8};

load Dataset.mat;

X=HOGValuesTEST;

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
clear C B;

X=reshape(X,size(X,1),4,20,20);

y=yTest;
% y=ones(length(y),1,20,20).*y;
% X=horzcat(X,y);

%%%%%%%%%%%%%%%%%

correct=0;

cnt=zeros(95,1);
corrCnt=zeros(95,1);

for i=1:size(X,1)
    x=X(1,:,:,:);
    x=reshape(x,4,20,[]);
    [pred,prob]=predict(x,filter1,filter2,weight3,weight4,bias1,bias2,bias3,bias4);
    cnt(y(i))=cnt(y(i))+1;
    if pred==y(i)
        correct=correct+1;
        corrCnt(pred)=corrCnt(pred)+1;
    end
    correct/(i)*100
end
fprintf('Total Accuracy = %.2f', (correct/size(X,1)*100));
        
