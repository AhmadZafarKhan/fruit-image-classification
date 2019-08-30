function output=conv(img,filterLayer,bias,stride)
%
% conv(image,filterLayer,bias,stride)
% 
% FilterLayer comes from initializeFilter. It should be a 4 dimensional
% matrix where 1st dim is the number of filters, 2nd dim is the number of z
% layers of the images or dataset and 3rd and 4th dim is the x or y dim.
%


%
% 
%
%
s=stride;
[nFilters,nChannelsF,yDimF,xDimF]=size(filterLayer); 

% We find the dimensions of the dataset
[nChannels, x_img, y_img]=size(img);

% We find the output dimensions after the convolution
outDim=floor((y_img - yDimF)/s) + 1;

% We carry out a basic check for dimension mismatch
if nChannels~=nChannelsF
    warning('Dimensions should match');
else
    % we initialize our output
%     output=zeros(nFilters,outDim,outDim);
    % We go through all the filters
    for currFilter=1:nFilters
        output_y=1;
        current_y=1;
        % We convolve the image with the filter vertically
        while current_y+yDimF <= y_img+s
            output_x=1;
            current_x=1;
            % We convolve the image with the filter horizontally
            while current_x+yDimF <= x_img+s
                %Main convolution operation
                output(currFilter,output_y,output_x)=sum(sum(sum(reshape(filterLayer(currFilter,:,:,:),nChannelsF,yDimF,xDimF).*img(:,[current_y:current_y+yDimF-1],[current_x:current_x+yDimF-1]),'omitnan'),'omitnan'),'omitnan') + bias(currFilter);
                current_x=current_x + s;
                output_x=output_x+1;
            end
        current_y=current_y+s;
        output_y=output_y+1;
        end
    end
end

end



