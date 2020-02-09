function output=softMax(predictions)
%
% Codes for the probability class since this is a classification problem
%

num=exp(predictions); % numerator value
output=num/(sum(num)); % denominator whic is sum of numerators equalling 1

end