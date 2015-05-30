function model = initializeHMMmodel(type,nstates,ostates)
    model.nstates = nstates;
    model.type = type;
    if strcmp(type,'discrete')
        model.ostates = ostates;
        model.pi = (1.0/nstates)*ones(nstates,1);
        model.A = rand(nstates,nstates);
        model.A = model.A./repmat(sum(model.A,2),1,nstates);
        model.B = (1.0/ostates)*ones(nstates,ostates);
    elseif strcmp(type,'gauss')
        model.observationDimension = ostates;
        model.pi = (1.0/nstates)*ones(nstates,1);
        model.A = rand(nstates,nstates);
        model.A = model.A./repmat(sum(model.A,2),1,nstates);
        for i = 1:model.nstates
            model.mu{i} = rand(model.observationDimension,1);
            model.sigma{i} = (1.0/model.observationDimension)*eye(model.observationDimension,model.observationDimension); % rbfK(rand(),model.observationDimension);
        end;
    end;
end

function K = rbfK(gam,s)
    K = zeros(s,s);
    for i = 1:s
        for j = 1:s
            K(i,j) = normpdf(i-j,0,gam);
        end;
    end;
    K = K + (1.0/s)*eye(s,s);
    
end

