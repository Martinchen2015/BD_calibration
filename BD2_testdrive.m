clear; clc; clf;
run('./manopt/importmanopt');      % make sure this is set properly!
addpath('./helpers');


%% *EITHER, generate A0 randomly:
%
k = [10 10];        % size of A0
n = 1;              % number of slices

% A0 = randn([k n]);
% A0 = normpdf(1:k(1), (k(1)+1)/2, k(1)/3)'*normpdf(1:k(2), (k(2)+1)/2, k(2)/3);
A0 = normpdf(1:k(1), (k(1)+1)/2, k(1)/2)'*normpdf(1:k(2), (k(2)+1)/2, k(2)/6);
 rot = imrotate(A0,45);
Apmap = zeros(20);
Apmap(5,5)=1;
Apmap(15,15)=1;
Ap = conv2(rot,Apmap);
Ap = Ap(5:29,5:29);
% Ap = imresize(Ap,1/2);
A0 = Ap;
A0 = imresize(A0,1/2);
k = size(A0);

A0 = A0/norm(A0(:));
%}

%% *Parameters to play around with:
m       = [50 50];    % size of x0 and Y
theta   = 3e-3;         % sparsity
eta     = 5e-3;         % additive Gaussian noise variance
%% generate Y

X0 = double(rand(m) <= theta);      % X0 ~ Bernoulli(theta)
bias = zeros(n,1);%randn(n,1);
Y = zeros([m n]);
for i = 1:n                         % convolve each slice
    Y(:,:,i) = cconvfft2(A0(:,:,i), X0) + bias(i);
end
Y = Y + sqrt(eta)*randn([m n]);     % add noise


%% Defaults for the options:
mu = 1e-6;              % Approximation quality of the sparsity promoter.
kplus = ceil(0.5*k);    % For Phase II/III: k2 = k + 2*kplus.
method = 'TR';          % Solver for optimizing over the sphere.
maxit = 20;             % Maximum number of iterations for the solver.
dispfun = ...           % the interface is a little wonky at the moment
    @( Y, a, X, k, kplus, idx ) showims( Y, A0, X0, reshape(a, [k n]), X, kplus, 1 );
dispfun1 = @(a, X) dispfun(Y, a, X, k, [], 1);
%% run the Manopt
Ain = randn([k n]); Ain = Ain/norm(Ain(:));
lambda1 = 0.1;
[A, Xsol, stats] = BD2_Cali_Manopt( Y, Ain, lambda1, mu, [], dispfun1, method, maxit);
%% display