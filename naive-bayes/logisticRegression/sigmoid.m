function g = sigmoid(z)

h = 1./(1 + exp(-z)); 

g = h;
end

