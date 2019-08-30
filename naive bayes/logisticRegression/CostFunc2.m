function [ J ] = CostFunc2( XTrain, yTrain, w )

    [n, nFeature] = size(XTrain);
    
    h = sigmoid(w*XTrain);
    size(h)
    size(yTrain)
    
    J = sum(-yTrain*log(h)-(1-yTrain)*log(1-h)); 
 
end
