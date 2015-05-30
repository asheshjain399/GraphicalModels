function model = hmmDemo( )
    clc
    model.type = 'gauss';
    
    %% Demo for discrete case
    if strcmp(model.type,'discrete')
        model.nstates = 4; 
        model.ostates = 6;
        model.B =[1/6  1/6   1/6   1/6   1/6   1/6  ;  
                 1/10  1/10  1/10  1/10  1/10  5/10 ; 
                 2/6   1/6   1/6   1/6    1/12  1/12 ;
                 7/12   1/12  1/12  1/12  1/12 1/12];  

        model.A = [0.6 0.15 0.20 0.05;
                  0.10 0.70 0.15 0.05;
                  0.10 0.30 0.10 0.50;
                  0.30 0.10 0.30 0.30];

        model.pi = [0.8 0.1 0.1 0]';
        model.type = 'discrete';
        T = 10;
        N = 100;
        [observations,hidden_states] = hmmSample(model,T,N);
        trueModel = model;
        model = initializeHMMmodel(model.type,model.nstates,model.ostates);
        hmmFit(trueModel,observations,model,300);
        
    elseif strcmp(model.type,'gauss')
        model.nstates = 4; 
        model.observationDimension = 10;
        model.A = [0.6 0.15 0.20 0.05;
                  0.10 0.70 0.15 0.05;
                  0.10 0.30 0.10 0.50;
                  0.30 0.10 0.30 0.30];
        for i = 1:model.nstates
            model.mu{i} = rand(model.observationDimension,1);
            model.sigma{i} = rbfK(rand(),model.observationDimension);
        end;
        model.pi = [0.8 0.1 0.1 0]';
        T = 10;
        N = 100;
        [observations,hidden_states] = hmmSample(model,T,N);
        model = initializeHMMmodel(model.type,model.nstates,model.observationDimension);
        
        model.prior.use = 1; % 1 use the prior, 0 not prior on parameters
        %% Prior values
        model.prior.k0 = 1;
        model.prior.mu0 = (1.0/model.observationDimension)*ones(model.observationDimension,1);
        model.prior.Psi = (1.0/model.observationDimension)*eye(model.observationDimension,model.observationDimension);
        truemodel = model;        
        hmmFit(truemodel,observations,model,300);
    end;

end

function K = rbfK(gam,s)
    K = zeros(s,s);
    for i = 1:s
        for j = 1:s
            K(i,j) = normpdf(i-j,0,gam);
        end;
    end;
    
end

