function [y] = shuffle(x)

[m n] = size(x); 
id = randperm(m); 
temp = x(id, :);      
size(temp)
y = temp; 

end

