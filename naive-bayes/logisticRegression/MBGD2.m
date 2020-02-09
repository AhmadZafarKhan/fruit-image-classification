
function [w] = MBGD2( xTrain, yTrain, w0, maxIter, learningRate)

[nSamples, nFeature] = size(xTrain);
batch_size = 64; 
w = w0;
precost = 0;
batch_size_count = 0; 
for j = 1:maxIter
    for k = 1:nSamples 
        batch_size_count = batch_size_count + 1; 
        temp = zeros(nFeature,1);
        count = randi(k); 
%         size(xTrain(count,:))
%         size(w)
        temp = temp + learningRate *((sigmoid(dot(xTrain(count,:), w)) - yTrain(count)))* xTrain(count,:)';
%         temp1 = temp1 + learningRate*temp;
%         temp2 = temp2 + learningRate*dot(temp, xTrain(count,:)');
        
        if batch_size_count == 64
            w = w - learningRate * temp/batch_size;
            batch_size_count = 0;
            temp = temp*0; 

        end
        
%         cost = CostFunc(xTrain, yTrain, w);
%         if j~=0 && abs(cost - precost) / cost <= 0.01
%             break;
%         end
%         precost = cost;
    end

end
