function evidence = calculateEvidence(model, data)
% This function calculates evidence for each observation for being emitted from states.
    evidence = cell(1,size(data,2));
    if strcmp(model.type,'gauss')
        for i = 1:size(data,2)
            evidence_sample = zeros(model.nstates,size(data{i},2));
            for j = 1:model.nstates
                mu = model.mu{j};
                sigma = model.sigma{j};
                probability = mvnpdf(data{i}',mu',sigma);
                evidence_sample(j,:) = probability';
            end;
            evidence{i} = evidence_sample;
        end;
    elseif strcmp(model.type,'discrete')
        for i = 1:size(data,2)
            evidence{i} = model.B(:,data{i});
        end;
    end;
end
