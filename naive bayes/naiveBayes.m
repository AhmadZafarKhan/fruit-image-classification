%% naive bayes classifier 
trainF = preProcesssing(HOGValuesTRAIN); 
testF = preProcesssing(HOGValuesTEST);

%%
trainF_n = [ones(length(trainF), 1), trainF];
testF_n = [ones(length(testF), 1), testF];
%% 
trainF_n2 = preProcesssing(HOGValuesTRAIN, 2); 
testF_n2 = preProcesssing(HOGValuesTEST, 2);
disp("done")
%% applying PCA
[trainF_n2_pca percent_vairance] = myPCA(trainF_n2, 400); 
[testF_n2_pca percent_vairance2] = myPCA(testF_n2, 400); 
disp("done") 

%% Naive bayes 2

%calculating mean and std dev of features
classes = 95; 
[c s] = pca(trainF_n2); 
s_new = s(:, 1:1000); 
trainF_n2_pca = s_new;

[c s] = pca(testF_n2); 
s_newT = s(:, 1:1000); 
testF_n2_pca = s_newT; 

xTrain = trainF_n2; 
xTest = testF_n2; 
[m n] = size(xTrain); 
piRatios = zeros(classes, 1); 
featIsolated = []; 
mu2 = zeros(classes, n);
stdDev2 = zeros(classes, n); 
featIsolated = zeros(size(xTrain(1, :))); 
for k = 1:classes
    piRatios(k) = sum(yTrain == k); 
end 


for i = 1:classes
    
    v = find(yTrain == i); 
    featIsolated = xTrain(v, :);
 
    mu2(i, :) = mean(featIsolated); 
    std_dev2(i, :) = sqrt(sum(abs(featIsolated - mu2(i, :)).^2)/length(v)); 
  
    
end
disp("learning of means and standard deviations is done") 


disp("Testing phase") 
y_mle = zeros(classes, 1); 
y_hat = zeros(classes, 1); 
probs = 0; 
[m, n] = size(xTest); 
for i = 1:m
    for j = 1:classes
        for k = 1:n
            sigma = std_dev2(j, k); 
            muu = mu2(j, k); 
            probs = probs + log(double(1/(sqrt(2*pi)*sigma)*exp(-((xTest(i, k)-muu).^2)./(2*sigma.^2)))); 
            if(isnan(probs))
                probs = 0;
            end
        end
        y_mle(j) = log(piRatios(j)) + (probs);
        probs = 0; 
%         disp(y_mle(j));
    end

    [D I] = max(y_mle);
    y_hat(i) = I; 
    y_mle = zeros(classes, 1); 
%     disp(i)

end
% disp("done"); 

disp("accuracy is: ")
error = sum(yTest ~= y_hat)
accuracy = 100*sum(yTest == y_hat)/length(yTest)

% conf= confusionmat(yTest', y_hat') 

%%
roc(yTest', y_hat')

