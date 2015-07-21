% testdriver for wolver2_pdNCG
clear; clc; clf;
run('../manopt/importmanopt');      % make sure this is set properly!
addpath('./helpers');

%% generate signal and set arguments
m = [256 256];
k = [49 49];
slice = 2;
theta = 5e-4;

bias = randn(slice,1);
A = zeros([k,slice]);
X = double(rand(m) <= theta) .* randn(m);
Y = zeros([m slice]);

for i = 1:slice
    A(:,:,i) = randn(k);
    tmp = A(:,:,i);
    A(:,:,i) = A(:,:,i)/norm(tmp(:));
    Y(:,:,i) = cconvfft2(A(:,:,i),X,m) + bias(i);
end

lambda = 0.1;
mu = 1e-6;

sol = wsolver2_pdNCG(Y, A, lambda, mu)
subplot(121);imagesc(abs(X));colorbar;subplot(122);imagesc(abs(sol.X));colorbar;