function [X] = preProcesssing(trainSet, l)

%train set is hogTrain{48905, 1}(20, 20, 4)

if l == 1
for i = 1:length(trainSet)
    x1 = [];
    x2 = []; 
    x3 = []; 
    x4 = [];
    y = []; 

    for j = 1:20        
        x1 = [x1 trainSet{i, 1}(j, :, 1)]; 
        x2 = [x2 trainSet{i, 1}(j, :, 2)];     
        x3 = [x3 trainSet{i, 1}(j, :, 3)]; 
        x4 = [x4 trainSet{i, 1}(j, :, 4)];    
    end
    y = [x1 x2 x3 x4];
    X(i, : ) = y; 
end
% disp(x1)
% disp(size(x1))
% disp(size(x2)) 
% disp(size(x3))
% disp(size(x4)) 
v = [x1 x2 x3 x4]; 


elseif l == 2
    
check = permute(trainSet, [4,3,2,1]); 
check2 = reshape(check, size(check, 1), []);
[m n] = size(check2); 
check2 = [ones(m, 1), check2]; 
X = check2; 
end

% X = v; 
end

