function [ res ] = ClassifyforN( XTest, w )

[m n] = size(w); 
y_mle = []; 
y_hat = []; 
nTest = size(XTest,1);
res = zeros(nTest,1);
for i = 1:nTest
    disp(i) 
    for j = 1:n 
        w0 = w(:, j); 
        y_mle = [ y_mle sigmoid(XTest(i,:) * w0) ];
    end
    [D I] = max(y_mle); 
%     disp([D I]); 
    y_hat(i) = I; 
    y_mle=[]; 

end
res = y_hat; 
end


