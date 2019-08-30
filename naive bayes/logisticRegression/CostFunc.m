function [ J ] = CostFunc( XTrain, yTrain, w )

    [n, nFeature] = size(XTrain);
    temp = 0;
    for m = 1:n
        h = sigmoid(XTrain(m,:) * w);
        if yTrain(m) == 1
            temp = temp + log(h);
        else
            temp = temp + log(1 - h);
        end
    end
    J = temp / (-n);
end
