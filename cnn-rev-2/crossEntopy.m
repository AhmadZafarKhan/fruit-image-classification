function y=crossEntopy(calcProb, label)
%
% We take the expected probablity values we estimated (calcProb) and the labels as the input
% We then return the loss.
%

output=sum(label.*log(calcProb));
y=-output;

end
