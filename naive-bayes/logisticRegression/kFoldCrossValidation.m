function [x0,xt0,y0,yt0, x1,xt1,y1,yt1, x2,xt2,y2,yt2, x3,xt3,y3,yt3, x4,xt4,y4,yt4] = ...
kFoldCrossValidation(xTrain,xTest,yTrain,yTest, K)
x = [xTrain; xTest]; 
y = [yTrain; yTest]; 

foldsize = floor(length(x)/K); 
f1 = x(1:foldsize, :); 
f2 = x(foldsize+1:foldsize*2, :); 
f3 = x(foldsize*2+1:foldsize*3, :);
f4 = x(foldsize*3+1:foldsize*4, :);
f5 = x(foldsize*4+1:foldsize*5, :);

yfold1 = y(1:foldsize, :); 
yfold2 = y(foldsize+1:foldsize*2, :); 
yfold3 = y(foldsize*2+1:foldsize*3, :);
yfold4 = y(foldsize*3+1:foldsize*4, :);
yfold5 = y(foldsize*4+1:foldsize*5, :);

%f1 f2  f3 f4 f5
x0 = [f2; f3; f4; f5];
xt0 = f1; 
y0 = [yfold2; yfold3; yfold4; yfold5];
yt0 = yfold1;

x1 = [f1; f3; f4; f5];
xt1 = f2; 
y1 = [yfold1 ;yfold3 ;yfold4 ;yfold5];
yt1 = yfold2; 

x2 = [f1; f2 ;f4 ;f5];
xt2 = f3; 

y2 = [yfold1; yfold2; yfold4 ;yfold5];
yt2 = yfold3; 

x3 = [f1; f2; f3; f5];
xt3 = f4; 
y3 = [yfold1; yfold2; yfold3; yfold5];
yt3 = yfold4; 

x4 = [f1; f2; f3; f4];
xt4 = f5; 
y4 = [yfold1; yfold2; yfold3; yfold4];
yt4 = yfold5; 

end

