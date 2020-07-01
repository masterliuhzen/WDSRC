function [id]= Classification_WDSRC(D,class_pinv_M,y,Dlabels)
%------------------------------------------------------------------------
% WDSR classification function
coef         =  class_pinv_M*y;
for ci = 1:max(Dlabels)
    coef_c   =  coef(Dlabels==ci);
    Dc       =  D(:,Dlabels==ci);
    error(ci) = norm(y-Dc*coef_c,2)^2;
%     error(ci) = norm(y-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
end

index      =  find(error==min(error));
id         =  index(1);
