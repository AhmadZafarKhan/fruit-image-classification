%% start of logistic regression script

%% PRE-PROCESSING
trainF = preProcesssing(HOGValuesTRAIN, 2); 
testF = preProcesssing(HOGValuesTEST, 2);
disp("done")

%% shrinking dataset to 20 classes
[trainF20, yTrain20] = shrink(trainF, yTrain);
[testF20, yTest20] = shrink(testF, yTest);
disp("done")
%%
[c s] = pca(trainF20); 
s_new = s(:, 1:400); 
trainF_n2_pca = s_new;

[c s] = pca(testF20); 
s_newT = s(:, 1:400); 
testF_n2_pca = s_newT; 
disp("done")
%%
tic 
xTrain = trainF_n2_pca; 
xTest = testF_n2_pca;
number_of_classes = 20; 
[n, p] = size(xTrain);
weight = zeros(p, number_of_classes);

for N = 1:number_of_classes
    y = zeros(size(yTrain20)); 
    k = find(yTrain20 == N);
    y(k) = 1; 

    w0 = zeros(p, 1);
    weight(:, N) = MBGD2( xTrain, y, w0, 200, 0.1);
    disp(N)    

end

y_hat = ClassifyforN( xTest, weight );
disp("done")

toc;
t = toc; 

err = sum(yTest20 ~= y_hat') 
accuracy = sum(yTest20==y_hat')/length(xTest);
disp("the accuracy in percentage is: "+accuracy*100)

conf= confusionmat(yTest20', y_hat')

%%
%logistic regression script

number_of_classes = 20; 
[n, p] = size(xTrain);
weight = zeros(p, number_of_classes);

for N = 1:number_of_classes
    y = zeros(size(yTrain20)); 
    k = find(yTrain20 == N);
    y(k) = 1; 
%     size(y)
    xTrain = trainF20; 
    xTest = testF20;
    weight = MBGD3( xTrain, y, weight, 200, 0.5);
    disp(N)    

end


y_hat = ClassifyforN( xTest, weight );
disp("done")


err = sum(yTest20 ~= y_hat') 
accuracy = sum(yTest20==y_hat')/length(xTest);
disp(accuracy*100)







