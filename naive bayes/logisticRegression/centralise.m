function [outputArg1] = centralise(inputArg1)

[m n] = size(inputArg1); 

for i = 1:n
    inputArg1(:, i) = (inputArg1(:, i) - mean(inputArg1(:, i))); 
end

outputArg1 = inputArg1;
end

