
function [w] = logisticRegressionWeights2( XTrain, yTrain, w_o, epoch, lr)

    [n, m] = size(XTrain);
    w = w_o;
    previous = 0;
    for j = 1:epoch
%         disp("iteration number: " + j)
        temp = zeros(m,1);
%         temp = zeros(m + 1,1);

        for k = 1:n
            temp = temp + (sigmoid([XTrain(k,:)] * w) - yTrain(k)) * [XTrain(k,:)]';
%             temp = temp + (sigmoid([1.0 XTrain(k,:)] * w) - yTrain(k)) * [1.0 XTrain(k,:)]';

        end
        w = w + lr * temp;
        cost = CostFunc(XTrain, yTrain, w);
        if j~=0 && abs(cost - previous) / cost <= 0.0001
            break;
        end
        previous = cost;
    end
% w = w/1601; 
% disp(size(w))
end


