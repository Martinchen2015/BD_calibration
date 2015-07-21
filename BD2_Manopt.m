function [ Aout, Xsol, stats ] = BD2_Manopt( Y, Ain, lambda, mu, varargin)
%BD2_MANOPT     BD using Manopt solvers.
%
%   [ Aout, Xsol, Stats ] = BD2_AS_Manopt( Y, Ain, Lambda, Mu ) solves
%   the BD problem given observation Y, initialization Ain, regularized 
%   using a pseudo-Huber penalty specified by quality Mu and weight Lambda. 
%
%   Returns the recovered kernel Aout, the activation map Xsol.X and its
%   dual variable Xsol.W, as well as a bunch of other variables in the 
%   Stats structure.
%
%
%   BD2_AS_Manopt( Y, Ain, lambda, mu, Xinit, Dispfun, Solver, Maxit )  
%   Xinit allows the user to provide a structure containing the fields 
%   Xinit.X and Xinit.W for initializing the primal and dual variables to
%   the pdNCG solver for X.
%
%   Dispfun allows the user to specify a display function handle for 
%   plotting figures, etc. at the end of each iteration. 
%
%   Solver allows the user to specify the Manopt solver used to do BD:
%       'TR' - Trust-region method. The default option.
%       'SD' - Steepest-descent method.
%       'CG' - Conjugate-gradient method.
%
%   Maxit allows the user to specify the maximum number of iterations.
%

    addpath('./helpers');
    k = size(Ain); 
    if (numel(k) > 2)
        n = k(3); k = k(1:2);
    else
        n = 1;
    end
    
    % Options for the x solver:
    INVTOL = 1e-6;
    INVIT = 2e2;
    
    % Options for Manopt solvers:
    options.verbosity = 2;
    options.tolgradnorm = 1e-4;
    options.linesearch = @linesearch;
    options.ls_contraction_factor = 0.2;
    options.ls_suff_decr = 1e-3;
    
    %% Handle the extra variables:    
    nvarargin = numel(varargin);
    if nvarargin > 4
        error('Too many input arguments.');
    end
    
    idx = 1;
    if nvarargin < idx || isempty(varargin{idx})
        suppack.xinit = xsolve2_pdNCG(Y, Ain, lambda, mu, [], INVTOL, INVIT);
    else
        suppack.xinit = varargin{idx};
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
    
    %% Set up the problem structure for Manopt and solve
    % The package containing supplement information for the cost, egrad,
    % and ehess functions:
    suppack.Y = Y;
    suppack.k = k;
    suppack.n = n;
    suppack.lambda = lambda;
    suppack.mu = mu;
    suppack.INVTOL = INVTOL;
    suppack.INVIT = INVIT;
    
    problem.M = spherefactory(prod(k)*n);
    problem.cost = @(a, store) costfun(a, store, suppack);
    problem.egrad = @(a, store) egradfun(a, store, suppack);
    problem.ehess = @(a, u, store) ehessfun(a, u, store, suppack);
    
    options.statsfun = @(problem, a, stats, store) statsfun( problem, a, stats, store, dispfun);
    %options.stopfun = @(problem, x, info, last) stopfun(problem, x, info, last, TRTOL);
    
    % Run Manopt solver:
    [Aout, stats.cost, ~, stats.options] = ManoptSolver(problem, Ain(:), options);
    Aout = reshape(Aout, [k n]);
    Xsol = xsolve2_pdNCG( Y, Aout, lambda, mu, suppack.xinit, INVTOL, INVIT );
    
end

function [ cost, store ] = costfun( a, store, suppack )
    if ~isfield(store, 'X')
        store = computeX( a, store, suppack );
    end
    
    cost = store.cost;
end

function [ egrad, store ] = egradfun( a, store, suppack )
    k = suppack.k;
    n = suppack.n;
    if ~isfield(store, 'X')
        store = computeX( a, store, suppack );
    end
    
        egrad = zeros(prod(k)*n,1);
    for i = 1:n
        idx = (i-1)*prod(k) + (1:prod(k));
        tmp = cconvfft2( store.X, cconvfft2( store.X, reshape(a(idx), k) ) - suppack.Y(:,:,i), [], 'left');
        tmp = tmp(1:k(1), 1:k(2));
        egrad(idx) = tmp(:);
    end
end

function [ ehess, store ] = ehessfun( a, u, store, suppack )
    k = suppack.k;
    n = suppack.n;
    if ~isfield(store, 'X')
        store = computeX( a, store, suppack );
    end

    ehess = H_function( u, suppack.Y, reshape(a, [k n]), store.X, ...
        suppack.lambda, suppack.mu , suppack.INVTOL, suppack.INVIT );
end

function [ store ] = computeX( a, store, suppack )
    % Updates the cache to store X*(A), and the active-set whenever a new
    % a new iteration by the trust-region method needs it.
    
    k = suppack.k; n = suppack.n;
    sol = xsolve2_pdNCG( suppack.Y, reshape(a, [k n]), suppack.lambda, ...
            suppack.mu, suppack.xinit, suppack.INVTOL, suppack.INVIT );
%    sol = xsolve2_newton( suppack.Y, reshape(a,suppack.k), suppack.lambda, ...
%            suppack.mu, suppack.xinit, suppack.INVTOL, suppack.INVIT );
    
    store.X = sol.X;
    % store.W = sol.W;
    store.cost = sol.f;
end

function [ stats ] = statsfun( problem, a, stats, store, dispfun) %#ok<INUSL>
    % stats.X = store.X;      % So X could be returned at the end.
    % stats.W = store.W;
    dispfun(a, store.X);
    pause(0.1);
end

