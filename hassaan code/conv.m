function y =conv(img,filter,bias,stride)
%
%conv(image,filter,bias,stride)
%

[x_f, y_f, z_f]=size(filter);

[x_img, y_img, z_img]=size(img);

outdim=floor(y_img - z_f)/s) + 1;

if x_img~=y_f
    return warning("Dimensions should match");
else
    output=zeros(x_f,outdim,outdim);
    for i=1:x_f
        output_y=0;
        current_y=output_y;
        while current_y +f <=y_img
            output_x=0;
            current_x=output_x;
            while current_x +f<=y_img
                %Main convolution operation
                output(i,output_y,output_x)=sum(filt(i)*image(:,[current_y:current_y+f],[current_x:current_x+f])) + bias(i);
                current_x=current_x + s;
                output_x=output_x+1;
            end
        current_y=current_y+s;
        output_y=output_y+1;
        end
    end
end

y=output;
end



