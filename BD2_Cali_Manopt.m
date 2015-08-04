function [ Aout, Wsol, stats] = BD2_Cali_Manopt(Y, Ain, lambda, mu, varargin)
% BD2_Cali_Manopt use manopt to solve the blind deconvolution problem with
% the calibration of measurement bias.
%
    addpath('./helpers');
    k = size(Ain);
    if numel(k) > 2
        n = k(3);
        k = k(1:2);
    else
        n = 1;
    end
    
    % Options for w solver
    INVTOL = 1e-8;
    INVIT = 2e2;
    
    %Options for manopt solvers
    options.verbosity = 2;
    options.tolgradnorm = 1e-8;
    options.linesearch = @linesearch;
    options.ls_contraction_factor = 0.2;
    options.ls_suff_decr = 1e-3;
    
    %% Handle extra variables
    nvarargin = numel(varargin);
    if nvarargin > 4
        error('Too many input arguments.');
    end
    
    idx = 1;
    if nvarargin < idx || isempty(varargin{idx})
        suppack.winit = wsolver2_pdNCG(Y, Ain, lambda, mu, [], INVTOL, INVIT);
        
    else
        suppack.winit = varargin{idx};
    end
    
    idx = 2;
    if nvarargin < idx || isempty(varargin{idx})
        dispfun = @(a,x) 0;
    else
        dispfun = varargin{idx};
    end
    
    idx = 3;
    if nvarargin < idx || isempty(varargin{idx}) || strcmp(varargin{idx}, 'TR')
        ManoptSolver = @trustregions;
    elseif strcmp(varargin{idx}, 'SD')
        ManoptSolver = @steepestdescent;
    elseif strcmp(varargin{idx}, 'CG')
        ManoptSolver = @conjugategradient;
    else
        error('Invalid solver option.')
    end
    
    idx = 4;
    if nvarargin >= idx && ~isempty(varargin{idx})
        options.maxiter = varargin{idx};
    end
    
    %% Set the problem for the manopt
    % we need some constant parameter which will be contained in the
    % 'suppack'. Also we needsome description of problem which are the 
    % manifold, gradient and hessian which will be contained in problme.
    suppack.Y = Y;
    suppack.k = k;
    suppack.n = n;
    suppack.lambda = lambda;
    suppack.mu = mu;
    suppack.INVTOL = INVTOL;
    suppack.INVIT = INVIT;
    
    problem.M = spherefactory(prod(k)*n);
    problem.cost = @(a,store) costfun(a, store,suppack);
    problem.egrad = @(a,store) egradfun(a, store, suppack);
    problem.ehess = @(a,u,store) ehessfun(a, u, store,suppack);

 
    options.statsfun = @(problem, a, stats, store) statsfun( problem, a, stats, store, dispfun);
    
    %% run the solver
    [Aout, stats.cost, ~, stats.options] = ManoptSolver(problem, Ain(:), options);
    Aout = reshape(Aout, [k n]);
    Wsol = wsolver2_pdNCG( Y, Aout, lambda, mu, suppack.winit, INVTOL, INVIT );
end

function [store] = computeW(a, store, suppack)
    % update the catch to store the W*(A)
    k = suppack.k;
    n = suppack.n;
    
    sol = wsolver2_pdNCG(suppack.Y,reshape(a,[k n]),suppack.lambda,suppack.mu, ...
        suppack.winit,suppack.INVTOL,suppack.INVIT);
    store.X = sol.X;
    store.beta = sol.beta;
    store.cost = sol.f;
end

function [stats] = statsfun(problem, a, stats, store, dispfun)
    dispfun(a,store.X);
    pause(0.1);
end

function [cost, store] = costfun(a, store, suppack)
    if ~(isfield(store,'X') && isfield(store,'beta'))
        store = computeW(a, store, suppack);
    end
    cost = store.cost;
end

function [egrad, store] = egradfun(a, store, suppack)
    if ~(isfield(store,'X') && isfield(store,'beta'))
        store = computeW(a, store, suppack);
    end
    
    k = suppack.k;
    n = suppack.n;
    egrad = zeros(prod(k)*n,1);
    for i = 1:n
        idx = (i-1)*prod(k) + (1:prod(k));
        g = cconvfft2(store.X, reshape(a(idx), k)) + store.beta(i) - suppack.Y(:,:,i);
        tmp = cconvfft2(store.X, g ,[], 'left');
        tmp = tmp(1:k(1),1:k(2));
        egrad(idx) = tmp(:);
    end
end

function [ehess, store] = ehessfun(a, u, store, suppack)
    if ~(isfield(store,'X') && isfield(store, 'beta'))
        store = computeW(a, store, suppack);
    end
    
    k = suppack.k;
    n = suppack.n;
    ehess = Haa_function(u, suppack.Y, reshape(a, [k,n]), store.X, store.beta, ...
        suppack.lambda, suppack.mu, suppack.INVTOL, suppack.INVIT);
end