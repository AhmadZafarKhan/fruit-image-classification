function [x_pca, percent_vairance] = myPCA(x, dimensions)

x = x(:, 2:end);
[m n] = size(x); 

x = x-mean(x); 
a = cov(x); 
[V D] = eig(a); 
c = sum(D); 
d = c(end-dimensions : end);
percent_vairance = sum(d)/sum(c);
v2 = fliplr(V);
v2 = v2(:, 1:dimensions); 

x_pca = x*v2; 



end

