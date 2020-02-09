function delOutput=backwardMaxpool(delPool,orig,filterSize,s)
%
% Maxpool backward convolution
%

[zAxisDim,origDim,]=orig;

delOutput=zeros(size(orig));

for zCurrent=1:zAxisDim
    yOutput=0;
    yCurrent=0;
    
    while yCurrent+filterSize<=origDim
        xOutput=0;
        xCurrent=0;
        
        while xCurrent+filterSize<=origDim
            A=orig(zCurrent,[yCurrent:yCurrent+filterSize],[xCurrent:xCurrent+filterSize]);
            [a,b]=find(A==max(A,[],'all'));
            delOutput(zCurrent,yCurrent+a,xCurrent+b)=delPool(zCurrent,yOutput,xOutput);
            xCurrent=xCurrent+s;
            xOutput=xOutput+1;
        end
        yCurrent=yCurrent+s;
        yOutput=yOutput+1;
    end
end
     