
function [w] = MBGD3( xTrain, yTrain, w0, maxIter, learningRate)

[nSamples, nFeature] = size(xTrain);
batch_size = 64; 
w = w0';
check = xTrain';
precost = 0;
batch_size_count = 0; 
for j = 1:maxIter
    temp = zeros(size(w0));
    for k = 1:nSamples 
        batch_size_count = batch_size_count + 1; 
        count = randi(k); 
%         xTrain(1:k, :) = xTrain(count, :); 
%         yTrain(1:k) = yTrain(count); 
%  disp(k); 
%         disp("size of check: ")
%         size(check) 
%         disp("size of yTrain: " )
%         size(yTrain)
%         disp("size of w: "  )
%         size(w)
%         ch = w*check; 
%         disp("size of ch: "  )
%         size(ch)
%         v = sigmoid(w*check); 
%         disp("size of v: " )
%         size(v)
%         activate = (v' - yTrain(k)); 
%         disp("size of activate: "  )
%         activate = activate';
%         size(activate)
        temp = temp + learningRate * (check*(((sigmoid(w*check))' ...
            - yTrain(k)))); 
%         disp("size of temp: " )
%         size(temp) 
        
        if batch_size_count == 64

            temp2 = temp'; 
            w = w - temp2/batch_size;
%             w = w ; 
            batch_size_count = 0;
%             size(w)
%             size(temp)
            temp = temp*0; 
            
        end
%         size(check)
        cost = CostFunc2(check, yTrain, w);
        if j~=0 && abs(cost - precost) / cost <= 0.01
            break;
        end
        precost = cost;
    end
    cost = 0; 
end
end
