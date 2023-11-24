function [Y] = whole_SVT(thr, raw, p, q)
% SVT算法，但要把raw矩阵的数据变换
[d, n] = size(raw);
% 每轮SVT计算，先把raw转为p*q的Q再求解
Y = zeros(d, n);
for i = [1:n]
    Q = raw(:,i);
    Q = reshape(Q,p,q);
    Yn = SVT(thr, Q);
    Y(:,i) = Yn(:);
end

end