function out=maxpool(image,f,s)

% f=2; % filter dimension size
% s=2; % step/stride size

[nChannels,yImg,xImg]=size(image);

outputHeight = floor((yImg - f)/s) +1;
outputWidth  = floor((xImg - f)/s) +1;

% initialize matrix for output
% downSampled = zeros(nChannels,outputHeight,outputWidth);

%For each z dimension

for i=1:nChannels
    yOutput=1;
    yCurrent=1;
    
    while (yCurrent +f <= yImg+s)
        % Sweep through Height
        xOutput=1;
        xCurrent=1;
        while (xCurrent+f<=xImg+s)
            % Sweep through width
            downSampled(i,yOutput,xOutput)=max(max(max(image(i,[yCurrent:yCurrent+f-1],[xCurrent:xCurrent+f-1]),[],'omitnan'),[],'omitnan'),[],'omitnan');
            
            % Update xCurrent
            xCurrent=xCurrent+s;
            xOutput=xOutput+1;
        end
        % Update yCurrent
        yCurrent=yCurrent +s;
        yOutput=yOutput+1;
    end
end
out=downSampled;
end
            
