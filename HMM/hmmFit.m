%% Implementation of HMM for both discrete and continuous output
% Author: Ashesh Jain
% Email: ashesh@cs.cornell.edu

function [model,ll] = hmmFit( truemodel, data, model, numiter )

% Input descriptions:
% model
%	type: 'discrete' or 'gauss'. Use 'discrete' for discrete observations and 'gauss' for continuous observations
%	nstates: dimension of state space
%	ostates: dimension of obeservation space when 'type' is 'discrete'
%	observationDimension: dimension of obeservation space when 'type' is 'gauss'
%	A: State transition matrix of dimension nstates x nstates
%	B: Observation emmision probability matrix when 'type' is 'discrete'. Dimension of B matrix is nstates x ostates
%	pi: Dimension is nstates x 1. This is the probabilty of states at T=0
%	mu: This is a cell of size nstates. It is used when 'type' is 'gauss'. 
%		mu{} is of dimension observationDimension x 1
%	sigma: This is a cell of size nstates. It is used when 'type' is 'gauss'
%		sigma{} is of dimension observationDimension x observationDimension	
%	prior
%		use: set this is 1 if you'd like to use prior over 'mu' and 'sigma'
%		mu0: This is prior over the mean 'mu'. This code currently use same prior for all 'mu{}'
%		Psi: This is prior over the mean 'sigma'. This code currently use same prior for all 'sigma{}'
%		k0: This is a scalar between [0,1]. Refer to Kevin Mruphy's book to understand the prior parameters.

% data
%	This is a cell of size equal to the number of hmm sequences
%	Each cell data{} stores the observations. 
%	When 'type'='discrete' then data{} is one dimensional array 
%	When 'type'='gauss' then data{} is a 2D array of size observationDimension x T. Where T is the length of HMM sequence

% truemodel
%	This is the ground truth model parameters. It is only used to check how close the learned model converges to truemodel in terms of likelihood values. Set this to dummy value if you don't know the ground truth. 

% numiter
%	Number of EM iterations.

% Output descriptions:
% model
%	Learned model parameters

% ll
%	Likelihood values at each iteration. Check this to make sure it becomes stable, and also make sure it does not decrease at any iteration. 

    ll = zeros(numiter,1);
    for i = 1:numiter
        evidence = calculateEvidence(model, data);
        [alph,bet,gam,xi,loglikelihood] = FBWpass(model, evidence);
        ll(i) = sum(cell2mat(loglikelihood));
        sufficient_statistics = estep(gam,xi,model,data);
        model = mstep(gam,sufficient_statistics,model,data);
    end;
    figure, plot(ll);
    %disp(sum(cell2mat(loglikelihood)));
    %[~,loglikelihood,~] = ForwardPass(truemodel, calculateEvidence(truemodel, data));
    %disp(sum(cell2mat(loglikelihood)));
end

function model = mstep(gam,expected,model,data)
    N = size(gam,2);
    model.pi = expected.start_state/N;
    model.A = expected.transition./repmat(sum(expected.transition,2),1,model.nstates);
    if strcmp(model.type ,'discrete')
        model.B = expected.observation_count./repmat(expected.state_frequency,1,model.ostates);
    elseif strcmp(model.type ,'gauss')
        for i = 1:model.nstates
            mu = expected.mean_vector(:,i)/expected.state_frequency(i);
            
            xmu = zeros(model.observationDimension,model.observationDimension);
            for x = 1:N
                lgamma = gam{x};
                gamma_times_data = repmat(lgamma(i,:),model.observationDimension,1).*data{x};
                xmu = xmu + gamma_times_data*repmat(mu',size(lgamma,2),1);
            end;
            
            
            %sigma = (expected.cov_matrix(:,:,i) - expected.state_frequency(i)*mu*mu')/expected.state_frequency(i);
            sigma = (expected.cov_matrix(:,:,i) + expected.state_frequency(i)*mu*mu' - xmu - xmu')/expected.state_frequency(i);
            if model.prior.use == 1
                c1 = model.prior.k0*expected.state_frequency(i)/(model.prior.k0 + expected.state_frequency(i));
                sigma = model.prior.Psi + sigma + c1*(model.prior.mu0-mu)*(model.prior.mu0-mu)';
                mu = (model.prior.k0*model.prior.mu0 + expected.state_frequency(i)*mu)/(model.prior.k0+expected.state_frequency(i));
            end;
            model.mu{i} = mu;
            model.sigma{i} = 0.5*(sigma + sigma');
        end;
    end;
end

function expected = estep(gam,xi,model,data)
    N = size(gam,2);
    expected.start_state = zeros(model.nstates,1);
    expected.transition = zeros(model.nstates,model.nstates);
    expected.state_frequency = zeros(model.nstates,1);
    for i = 1:N
        lgamma = gam{i};
        expected.start_state = expected.start_state + lgamma(:,1);
        expected.transition = expected.transition + sum(xi{i},3);
        expected.state_frequency = expected.state_frequency + sum(lgamma,2);
    end;
    
    if strcmp(model.type,'gauss')
        expected.mean_vector = zeros(model.observationDimension,model.nstates);
        expected.cov_matrix = zeros(model.observationDimension,model.observationDimension,model.nstates);
        for i = 1:N
            lgamma = gam{i};
            for j = 1:model.nstates
                gamma_times_data = repmat(lgamma(j,:),model.observationDimension,1).*data{i};
                expected.mean_vector(:,j) = expected.mean_vector(:,j) + sum(gamma_times_data,2);
                expected.cov_matrix(:,:,j) = expected.cov_matrix(:,:,j) + gamma_times_data*data{i}';
            end;
        end;
    elseif strcmp(model.type,'discrete')
        expected.observation_count = zeros(model.nstates,model.ostates);
        for i = 1:N
            lgamma = gam{i};
            for j = 1:model.ostates
                expected.observation_count(:,j) = expected.observation_count(:,j) + sum(lgamma(:,find(data{i}==j)),2);
            end;
        end;
    end;
end

function [alph,bet,gam,xi,loglikelihood] = FBWpass(model, evidence)
    [alph,loglikelihood,mult_const] = ForwardPass(model, evidence);
    bet = BackwardPass(model, evidence,mult_const);
    gam = cell(1,size(evidence,2));
    xi = cell(1,size(evidence,2));
    for i = 1:size(evidence,2)
        lgam = alph{i}.*bet{i};
        lgam = lgam./repmat(mult_const{i},model.nstates,1);
        gam{i} = lgam;
    end;
    
    for i = 1:size(evidence,2)
        levidence = evidence{i};
        xi_sample = zeros(model.nstates,model.nstates,size(levidence,2)-1);
        for j = 1:size(levidence,2)-1
            xi_sample(:,:,j) = model.A.*(alph{i}(:,j)*(levidence(:,j+1).*bet{i}(:,j+1))');
            val = sum(sum(xi_sample(:,:,j)));
            
            assert(abs(val-1.0)<1e-6);
        end;
        xi{i} = xi_sample;
    end;
end

function [alph,loglikelihood,mult_const] = ForwardPass(model, evidence)
    alph = cell(1,size(evidence,2));
    loglikelihood = cell(1,size(evidence,2));    
    mult_const = cell(1,size(evidence,2));    
    for i = 1:size(evidence,2)
        levidence = evidence{i};
        alpha_sample = zeros(model.nstates,size(levidence,2));
        const = zeros(1,size(levidence,2));
        [alpha_sample(:,1), Z] = normalize_local(levidence(:,1).*model.pi);
        Zlog = log(Z);
        const(1) = 1.0/Z;
        for j = 2:size(levidence,2)
            [alpha_sample(:,j), Z] = normalize_local(levidence(:,j).*(model.A'*alpha_sample(:,j-1)));
            Zlog = Zlog + log(Z);
            const(j) = 1.0/Z;
        end;
        alph{i} = alpha_sample;
        loglikelihood{i} = Zlog;
        mult_const{i} = const;
    end;
end

function bet = BackwardPass(model, evidence, mult_const)
    bet = cell(1,size(evidence,2));
    for i = 1:size(evidence,2)
        levidence = evidence{i};
        const = mult_const{i};
        beta_sample = zeros(model.nstates,size(levidence,2));
        beta_sample(:,end) = 1;
        beta_sample(:,end) = const(end)*beta_sample(:,end);
        for j = (size(levidence,2)-1):-1:1
            beta_sample(:,j) = model.A*(levidence(:,j+1).*beta_sample(:,j+1));
            beta_sample(:,j) = const(j)*beta_sample(:,j);
        end;
        bet{i} = beta_sample;
    end;
end

function [v,Z] = normalize_local(v)
    Z = sum(v);
    v = v/Z;
end
