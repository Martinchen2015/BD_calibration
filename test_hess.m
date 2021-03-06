addpath('./helpers');
%w = ones(prod(m)+1,1);
w = randn(prod(m)+1,1);
%X = w(1:prod(m));
X = zeros(prod(m),1);
X_dual = X;
D = 1./sqrt(mu^2 + X(:).^2);
Hdiag = lambda*D.*(1 - D.*X(:).*X_dual(:));
Hdiag = zeros(size(Hdiag));
result_p = hess*w + [Hdiag;0].*w;
result_f = Hww_function(w, RA_hat, Hdiag, A_sum);