function [X] = preProcesssing(trainSet, l)

if l == 1
for i = 1:length(trainSet)
    x1 = [];
    y = []; 

    for j = 1:64        
        x1 = [x1 trainSet{i, 1}(j, :)];  
    end
    y = [x1 x2 x3 x4];
    X(i, : ) = y; 
end

v = [x1 x2 x3 x4]; 


elseif l == 2
    
check = permute(trainSet, [4,3,2,1]); 
check2 = reshape(check, size(check, 1), []);
[m n] = size(check2); 
check2 = [ones(m, 1), check2]; 
X = check2;

elseif l == 3
    
check = permute(trainSet, [4,3,2,1]); 
check2 = reshape(check, size(check, 1), []);
[m n] = size(check2); 
X = check2; 
end

% X = v; 
end

