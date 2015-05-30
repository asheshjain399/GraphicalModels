function [observations,hidden_states] = hmmSample( model,T,Nobs )

    S = markovSample(model, T, Nobs);
    observations = cell(1,Nobs);
    hidden_states = cell(1,Nobs);
    
    if strcmp(model.type,'discrete')
        for i = 1:Nobs
            hidden_states{i} = S(i,:);
            observations{i} = zeros(1, T);
            for t=1:T
                observations{i}(t) = sampleDiscrete(model.B(hidden_states{i}(t), :));
            end;
        end;
    elseif strcmp(model.type,'gauss')
        for i = 1:Nobs
            hidden_states{i} = S(i,:);
            observations{i} = zeros(model.observationDimension, T);
            for t=1:T
                k = hidden_states{i}(t); 
                observations{i}(:, t) = colvec(gaussSample(model.mu{k}, model.sigma{k}, 1));
            end
        end;        
    end;



end

