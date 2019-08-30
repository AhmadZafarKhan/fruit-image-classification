function [ res ] = classifyforN2( XTest, w )

    size(w)
    nTest = size(XTest,1);
    res = zeros(nTest,1);
    for i = 1:nTest
        for j = 1:95
            sigm = sigmoid(XTest(i,:) * w(j, :)');
            if sigm >= 0.5
                res(i) = j;
            else
                res(i) = 0;
            end
        end
    end
end
