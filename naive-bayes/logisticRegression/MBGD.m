
function [w] = MBGD( xTrain, yTrain, w0, maxIter, learningRate)

    [nSamples, nFeature] = size(xTrain);
    batch_size = 64; 
    w = w0;
    size(w0)

    precost = 0;

    for j = 1:maxIter
%         disp(j)
        temp = zeros(nFeature,1);
        temp1 = zeros(nFeature,1);

            for i = 1:batch_size
                count = randi(length(xTrain) ); 
%                 temp = temp + learningRate *((sigmoid(dot(xTrain(j,:), w)) - yTrain(j)) * xTrain(j,:)');
                temp = temp + learningRate *((sigmoid(dot(xTrain(count,:), w)) - yTrain(count)));
                temp1 = temp1 + learningRate*dot(temp, xTrain(count,:)');
            end

%     end
        temp = temp/length(temp);

        w = w + learningRate * temp;

        cost = CostFunc(xTrain, yTrain, w);

        if j~=0 && abs(cost - precost) / cost <= 0.01

            break;

        end

        precost = cost;

    end
end
