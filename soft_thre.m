function [Y] = soft_thre(b, lambda)
% 软阈值(Soft Thresholding)
    Y = sign(b).*max(abs(b) - lambda, 0);
end