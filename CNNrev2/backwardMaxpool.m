function delOutput=backwardMaxpool(delPool,orig,filterSize,s)
%
% Maxpool backward convolution
%

[nChannels,origDim,~]=size(orig);

delOutput=zeros(size(orig));

for zCurrent=1:nChannels
    yOutput=1;
    yCurrent=1;
    
    while yCurrent+filterSize<=origDim+1
        xOutput=1;
        xCurrent=1;
        
        while xCurrent+filterSize<=origDim+1
            A=orig(zCurrent,[yCurrent:yCurrent+filterSize-1],[xCurrent:xCurrent+filterSize-1]);
            %%%%%%%%%%%%%%%%%
            %
            % This is the problematic code
            %
            A=reshape(A,2,2);
            [a,b]=find(A==max(max(max(A,[],'omitnan'),[],'omitnan'),[],'omitnan'),1);
            %%%%%%%%%%%%%%%%%%
            delOutput(zCurrent,yCurrent+a-1,xCurrent+b-1)=delPool(zCurrent,yOutput,xOutput);
            
            xCurrent=xCurrent+s;
            xOutput=xOutput+1;
        end
        yCurrent=yCurrent+s;
        yOutput=yOutput+1;
    end
end
end

     