function [x0, y0] = shrink(x, y)

k = find(y<=20); 
y0 = y(k); 
x0 = x(k, :); 
end

