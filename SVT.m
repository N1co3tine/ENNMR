function [Y] = SVT(thr, Q)
% 奇异值阈值收缩算法
% Q = U*diag()*V'
[U, S, V] = svd(Q,"econ");
[~,r] = size(S);

temp = zeros(r,r);
for i = [1:r]
    if (S(i,i) - thr>0)
        temp(i,i) = S(i,i) - thr;
    end
end
Y = U * temp * V';

end