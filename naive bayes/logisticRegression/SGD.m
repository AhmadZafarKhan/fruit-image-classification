function [g] = SGD(x,b, y)
% 
% [a n] = size(x); 
% xt = x'; 
% lr = 0.01; 
% m = length(y); 
% output = dot(x, b); 
% guess = sigmoid(output); 
% t = zeros(1, n); 
% slope = zeros(1, n); 
%     
% for i = 1:n
% 
%     error = guess - y(1); 
%     t(i) = t(i) + error*x(i); 
% 
% end
% 
% slope = t/n; 
% 
% for i = 1:n
% %     slope = SGD(x, b, y);   
%     b(i) = b(i) - ...
%         lr*slope(i); 
% end
% weight = b; 

% disp(size(x))
% disp(size(b))
[m n] = size(x); 
lr = 0.0001; 
epsilon = 0.001; 

for i = 1:10
    z = dot(x, b); 
    a = sigmoid(z); 
    % disp("this is a: " + a)
    % disp(size(b))
    b_old = b; 
    b = b_old - lr*x.*(a-y); 
    
    if (abs(b-b_old) < epsilon)
        break
    end    
%     disp(i)
end

g = b;
end

