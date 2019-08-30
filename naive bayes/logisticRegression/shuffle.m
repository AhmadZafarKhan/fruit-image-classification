function [y] = shuffle(x)

[m n] = size(x); 
id = randperm(m); 
temp = x(id, :);      
y = temp; 

end

