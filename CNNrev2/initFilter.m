function y=initFilter(filterParam)
%
% filter has dims : x-y-z-nF
% x=y dims and nF is the no of Filters
%

scale=1;

standDev=scale/sqrt(prod(prod(prod(filterParam,'omitnan'),'omitnan'),'omitnan'));

y=normrnd(0,standDev,[filterParam]);

end





