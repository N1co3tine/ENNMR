function [X,Y] = ENNMR(A, B, Class_NUM, lambda, beta, p, q, mu, DEBUG)
%  解决以下优化问题
%  min_(||Y||_* + 1/2 * lambda * ||x||_2 ^ 2 + beta * ||w||_1
%  + Tr(Z1' * (A(x) - Y - B)) + Tr(Z2' * (x-w)) 
%  + mu/2 * ||A(x) - Y - B||_F ^ 2 + mu/2 * ||x-w||_2*2
%  mu:拉格朗日惩罚x系数， lambda:岭回归惩罚系数， beta:拉索回归惩罚系数
error_abs = 1e-5;
error_rel = 1e-5;

% d = p*q A: (p*q)*m | B: (p*q)*n  |   
[d, m] = size(A); %m为基底数量
[d, n] = size(B); %n为测试样本数
train_each = m/Class_NUM;
test_each = n/Class_NUM;

%B = B./256;

%% initial parameters
% compute M, H((p*q)*M) is A
M = inv(A'*A + (lambda./mu).*eye(m)); % m*pq
%X = ones(m,1)./m;
X = zeros(m,1);
W = X;
Y = -B;
Z1 = zeros(d, n);
Z2 = zeros(m, n);
correct = 0;

%% iteration,由于每个测试样本结束条件判定不一样，每个测试样本单独迭代
%parpool(8);
for i = [1:n]
    Bi = B(:,i);
    Yi = Y(:,i);
    Z1i = Z1(:,i);
    Z2i = Z2(:,i);
    Wi = zeros(m,1);
    k = 0;
    while(1)
        k = k + 1;
        Yi_pre = Yi;
        % update X
        g = A'*(Bi + Yi - (1/mu).*Z1i) + Wi - (1/mu)*Z2i;
        Xi = M*g;
        % update Y, but the second need to be changed to p*q*n
        Yi = whole_SVT(1/mu, A*Xi - Bi + (1/mu).*Z1i, p, q);
        % updage W
        Wi = soft_thre(Xi+(1/mu).*Z2i,beta/mu);
        r_pri1 = A*Xi - Yi - Bi;
        Z1i = Z1i + mu * r_pri1;
        r_pri2 = Xi - Wi;
        Z2i = Z2i + mu * r_pri2;
        
         % 判断结束条件
        error_pri = error_rel * max(norm(A*Xi,2),max(norm(Bi,2),norm(Yi,2)));
        % 先判断终止条件一是否满足，节省条件二的运算
        if (norm(r_pri1, 2) <= error_pri)
            if (norm(r_pri2, inf) <= error_rel*norm(Xi,inf))
                %error_dual = sqrt(m) * error_abs + error_rel * norm(A'*Z1i,2);
                %s_dual = mu*A'*(Yi-Yi_pre);
                if(norm(r_pri1, inf) <= error_rel*norm(Yi,inf))
                    if (DEBUG==1)
                        imshow(uint8(reshape(Bi, p, q)));
                        img = reshape(A*Xi, p, q);
                        imshow(uint8(img));
                        [~,max_idx] = max(Xi);
                        error = reshape(Yi, [p, q]);
                        Singular_Value_V = svd(error);
                        sum(abs(Singular_Value_V))
                        
                        lambda*norm(Xi,2)
                        beta*norm(Xi,1)
                        mu/2 * norm(A*Xi - Yi - Bi,2)
                        disp(max_idx);
                    end
                    break
                end
            end
        end
        
        if (DEBUG==1 && mod(k, 100) == 0)
            disp(k);
            %break;
        end
    end
    
    Class_Reconstruction_Error_V_Nu = zeros(Class_NUM,1);
    % 判别阶段
    for t = 1 : Class_NUM
       %X为基底中所有该分类的照片
       A_class       = A(:,(t-1) * train_each + 1 : t * train_each);
       %Class_W为基底中所有该分类的系数
       X_class = Xi((t-1) * train_each + 1 : t * train_each);
       %加权形成一个该分类组成的图像
       Reconstruction_Test_Sample = A_class * X_class;
       % Differ = Test(:)-Reconstruction_Test_Sample;
       % Ax is the reconstruction Image of the test image
       %取Ax与分类组合图像的差，计算这个残差矩阵的核范数
       Differ           = A*Xi - Reconstruction_Test_Sample;     
       Differ_Mat       = reshape(Differ, [p, q]);
       Singular_Value_V = svd(Differ_Mat);    
       
       Class_Reconstruction_Error_V_Nu(t) = sum(abs(Singular_Value_V));
       
    end
    %核范数最小的那个样本就是与Ax最类似的，就是人脸识别的结果
    [Min_Error,Class_No_Nu] = min(Class_Reconstruction_Error_V_Nu);
      
    if Class_No_Nu==fix((i-1)/test_each)+1 % strncmp is to compare the first n characters of two strings
        correct = correct + 1;
        disp([num2str(i), ' is correct']) 
    end
end
rate = correct / n

end