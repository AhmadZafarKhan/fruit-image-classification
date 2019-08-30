function [delOutput, delFilter, delBias]=backwardConvolution(delPrevConv, inputConv, filter, s);
%
% Backward convolution
%

[numberFilters,nChannelsF,filtSize,filtSize]=size(filter); % after the filter is initialized
[nChannelsConv,yInputConv,xInputConv]=size(inputConv);

delOutput=zeros(size(inputConv));
delFilter=zeros(size(filter));
delBias=zeros(numberFilters,1);

for currentFilt=1:numberFilters
    %Through all filters
    yOutput=1;
    yCurrent=1;
    
    while yCurrent+filtSize<=yInputConv+s
        xOutput=1;
        xCurrent=1;
        while xCurrent+filtSize<=yInputConv+s %y and x are equal for input conv
            %gradient used to update the filter
            delFilter(currentFilt,:,:,:)=reshape(delFilter(currentFilt,:,:,:),nChannelsF,filtSize,filtSize)+(delPrevConv(currentFilt,yOutput,xOutput)*inputConv(:,[yCurrent:yCurrent+filtSize-1],[xCurrent:xCurrent+filtSize-1]));
            %loss gradient of input to conv layer
            delOutput(:,[yCurrent:yCurrent+filtSize-1],[xCurrent:xCurrent+filtSize-1]) =delOutput(:,[yCurrent:yCurrent+filtSize-1],[xCurrent:xCurrent+filtSize-1]) + (delPrevConv(currentFilt,yOutput,xOutput)*reshape(delFilter(currentFilt,:,:,:),nChannelsF,filtSize,filtSize));
            xCurrent=xCurrent + s;
            xOutput=xOutput+1;
        end
        yCurrent=yCurrent + s;
        yOutput=yOutput+1;
    end
    delBias(currentFilt)=sum(sum(sum(delPrevConv(currentFilt,:,:),'omitnan'),'omitnan'),'omitnan');
end
end
