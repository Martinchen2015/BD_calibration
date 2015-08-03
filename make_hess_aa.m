% make Hess_aa to debug
% we need A, Y, Xsol.X, Xsol.beta, lambda1

%% make aa
C_x = circmtx2(Xsol.X);
eye_k_m = [eye(k);zeros(m(1)-k(1),k(1))];
eye_kk_mm = zeros(prod(m),prod(k));
for i = 1:k(1)
    idx_m = (i-1)*m(1) + (1:m(1));
    idx_k = (i-1)*k(1) + (1:k(1));
    eye_kk_mm(idx_m,idx_k) = eye_k_m;
end
tmp = C_x * eye_kk_mm;
Hess_aa = (tmp)' * tmp;

%% make aw
flip_fun = circshift(flip(eye(m)),1);
flip_fun_2D = cell(m);
for i = 1:m(1)
    for j = 1:m(2)
        flip_fun_2D{i,j} = zeros(m);
    end
end
for i = 1:m(1)
    flip_fun_2D{i,i} = flip_fun;
end
flip_fun_2D = cell2mat(circshift(flip(flip_fun_2D),1));
left_1 = circmtx2(cconvfft2(A,Xsol.X,m,'right'));
left_2 = circmtx2(cconvfft2(A,Xsol.X)+Xsol.beta-Y);
%left_2 = circmtx2(cconvfft2(A,Xsol.X)-Y);
left = left_1 + left_2 * flip_fun_2D;% + Xsol.beta;
right = C_x' * ones(prod(m),1);
Hess_aw = [left right];
Hess_aw = eye_kk_mm' * Hess_aw;

%make wa
left_1 = circmtx2(cconvfft2(A,Xsol.X,m,'left'));
left_2 = circmtx2(cconvfft2(A,Xsol.X)+Xsol.beta-Y);
right = right';
left = left_1 + left_2 * flip_fun_2D;
Hess_wa = [left;right];
Hess_wa = Hess_wa * eye_kk_mm;

%% calculate ww
hesspendiag = lambda1 * mu^2*(mu^2 + Xsol.X(:).^2).^(-3/2);
A_p = zeros(m);
A_p(1:k(1),1:k(2)) = A;
tmp = [circmtx2(A_p) ones(prod(m),1)];
Hess_ww = tmp' * tmp;
Hess_ww(1:prod(m),1:prod(m)) = Hess_ww(1:prod(m),1:prod(m)) + diag(hesspendiag);

%% make hessian
Hess_phy = Hess_aa - Hess_aw * inv(Hess_ww) * Hess_wa;