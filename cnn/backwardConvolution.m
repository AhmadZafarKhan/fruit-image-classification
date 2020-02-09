function [delOutput, delFilter, delBias]=backwardConvolution(delPrevConv, inputConv, filter, s);
%
% Backward convolution
%

[nF,imgDepth,FiltSize,FiltSize]=filter;
% [nFilt,yFilt,zFilt]=size(filter);
(,yInputConv,)=size(inputConv);

delOutput=zeros(size(inputConv));
delFilter=zeros(size(filter));
delBias=zeros(size(xFilt,1);

for currentFilt=1:xFilter
    %Through all filters
    yOutput=0;
    yCurrent=0;
    
    while yCurrent+f<=yInputConv;
        xOutput=0;
        xCurrent=0;
        while xCurrent+f<=yInputConv; %y and x are equal for input conv
            %gradient used to update the filter
            delFilter(currentFilt)=delFilter(currentFilter)+(delPrevConv(currentFilt,yOutput,xOutput)*inputConv(:,[yCurrent:yCurrent+FiltSize],[xCurrent:xCurrent+FiltSize]));
            %loss gradient of input to conv layer
            delOutput(:,[yCurrent:yCurrent +FiltSize],[xCurrent:xCurrent+FiltSize]=delOutput(:,[yCurrent:yCurrent +FiltSize],[xCurrent:xCurrent+FiltSize]) + (delPrevConv(CurrentFilt,yOutput,xOutput)*filter(currentFilter));
            xCurrent=xCurrent + s;
            xOutput=xOutput+1;
        end
        yCurrent=yCurrent + s;
        yOutput=yOutput+1;
    end
    delBias(currentFilter)=sum(delPrevConv(currentFilter));
end
