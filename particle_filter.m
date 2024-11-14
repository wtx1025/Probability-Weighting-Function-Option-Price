function loglikelihood = particle_filter(params, data)
    % params1, params2: alpha, beta 
    % params3, params4: phi_alpha, phi_beta 
    % params5, params6: variance_alpha, variance_beta 
    % params7, params8, params9: mu, sigma, gamma  

    % some pre process
    rng = 0; 
    mu = params(7);
    sigma = params(8); 
    gamma = params(9); 
    lowerBound = -0.5;
    upperBound = 0.8;
    numGrids = 200;
    gridValues = linspace(lowerBound, upperBound, numGrids); 

    x0Guess = [params(1); params(2)] + [0.01; -0.05];
    [xGrid, yGrid] = meshgrid(x0Guess(1)-0.4:0.05:x0Guess(1)+0.4, x0Guess(2)-0.2:0.05:x0Guess(2)+0.2);
    xVec = xGrid(:);
    yVec = yGrid(:); 
    states = [xVec, yVec]';
    [dim1, numberParticle] = size(states); 
    weights = (1 / numberParticle) * ones(1, numberParticle); 
    iterations = 702; 
    stateList = {};
    stateList{end+1} = states;
    weightList = {};
    weightList{end+1} = weights; 
    loglikelihood = 0; 

    % define state space model 
    A = [params(3), 0; 0, params(4)];
    processMean = [0; 0];
    processCov = [params(5), 0; 0, params(6)];
    observationMean = 0;
    observationCov = 800; 

    % particle filter algorithm 
    for i = 1 : iterations

        newStates = A * states + mvnrnd(processMean, processCov, numberParticle)';
        newWeights = zeros(1, numberParticle); 
        
        for j = 1:numberParticle

            % Rebound mechanism 
            if newStates(1, j) < 0.4
                rebound_distance = 0.4 - newStates(1, j);
                newStates(1, j) = newStates(1, j) + 3 * rebound_distance; 
            elseif newStates(1, j) > 1.6 
                rebound_distance = newStates(1, j) - 1.6; 
                newStates(1, j) = newStates(1, j) - 3 * rebound_distance; 
            end
    
            if newStates(2, j) < 0.6
                newStates(2, j) = max(newStates(2, j), 0.5); % prevent the state update to extreme value 
                rebound_distance = 0.6 - newStates(2, j);
                newStates(2, j) = newStates(2, j) + 3 * rebound_distance; 
            elseif newStates(2, j) > 1.4 
                newStates(2, j) = min(newStates(2, j), 1.6); % prevent the state update to extreme value 
                rebound_distance = newStates(2, j) - 1.4; 
                newStates(2, j) = newStates(2, j) - 3 * rebound_distance; 
            end 
        
            temp_numerator = 0;
            temp_denominator = 0; 
            for k = 1 : length(gridValues)
                currentValue = gridValues(k); 
                stock_price = data{i, 'spindx'}; 
                option_payoff = max(stock_price*(1+currentValue)-data{i, 'strike_price'}/1000, 0);

                c1 = 0;
                c2 = 0;
                if k==1
                    c2 = normpdf(currentValue, mu, sigma) * 0.0065;
                    c1 = 0;
                else
                    for s = 1 : k-1
                        c1 = c1 + normpdf(gridValues(s), mu, sigma) * 0.0065;
                    end
                    for t = 1 : k
                        c2 = c2 + normpdf(gridValues(t), mu, sigma) * 0.0065;
                    end
                end

                temp_numerator = temp_numerator + option_payoff*(1+currentValue)^(-gamma)...
                    *(exp(-(-newStates(2,j)*log(c2))^newStates(1,j))...
                    - exp(-(-newStates(2,j)*log(c1))^newStates(1,j)));
                temp_denominator = temp_denominator + (1+data{i, 'DTB3'})*(1+currentValue)^(-gamma)...
                *(exp(-(-newStates(2,j)*log(c2))^newStates(1,j))...
                - exp(-(-newStates(2,j)*log(c1))^newStates(1,j)));
            end

            meanDis = temp_numerator / temp_denominator; 
            distribution0 = mvnpdf(data{i, 'mid_quotes'}, meanDis, observationCov);
            newWeights(j) = real(distribution0 * weights(j)); 
            %fprintf('i=%d j=%d meanDis=%.4f observe=%.2f prob=%.4f weight=%.6f\n', i, j, meanDis, data{i, 'mid_quotes'},...
            %    distribution0, newWeights(j));

        end 
        
        weightStandardized = real(newWeights / sum(newWeights));
        tmp1 = weightStandardized.^2;
        Neff = 1 / sum(tmp1);
        if Neff <(numberParticle / 3)
            resampleStateIndex = randsample(1:numberParticle, numberParticle, true, weightStandardized);
            newStates = newStates(:, resampleStateIndex); 
            weightStandardized = (1 / numberParticle) * ones(1, numberParticle);
        end 

        states = newStates;
        weights = weightStandardized;
        stateList{end+1} = states;
        weightList{end+1} = weights; 
        
        % compute likelihood 
        current_states = stateList{end};
        current_weights = weightList{end};
        estimated_alpha = current_states(1, :) * current_weights';
        estimated_beta = current_states(2, :) * current_weights'; 
        numerator = 0;
        denominator = 0; 
        parfor k = 1 : length(gridValues)
            currentValue = gridValues(k); 
            stock_price = data{i, 'spindx'}; 
            option_payoff = max(stock_price*(1+currentValue)-data{i, 'strike_price'}/1000, 0);

            c1 = 0;
            c2 = 0;
            if k==1
                c2 = normpdf(currentValue, mu, sigma) * 0.0065;
                c1 = 0;
            else
                for s = 1 : k-1
                    c1 = c1 + normpdf(currentValue, mu, sigma) * 0.0065;
                end
                for t = 1 : k
                    c2 = c2 + normpdf(currentValue, mu, sigma) * 0.0065;
                end
            end

            numerator = numerator + option_payoff*(1+currentValue)^(-gamma)...
                *(exp(-(-estimated_beta*log(c2))^estimated_alpha)...
                - exp(-(-estimated_beta*log(c1))^estimated_alpha));
            denominator = denominator + (1+data{i, 'DTB3'})*(1+currentValue)^(-gamma)...
            *(exp(-(-estimated_beta*log(c2))^estimated_alpha)...
            - exp(-(-estimated_beta*log(c1))^estimated_alpha));
        end

        model_implied_price = numerator / denominator;
        loglikelihood = loglikelihood + log(mvnpdf(data{i, 'mid_quotes'}, model_implied_price, observationCov));
     

        fprintf(['Iteration %d done! alpha=%.2f beta=%.2f theoretical_price=%.2f observed_price=%.2f'...
            ' Loglikelihood = %.4f\n'], i, estimated_alpha, estimated_beta, model_implied_price,...
            data{i, 'mid_quotes'}, loglikelihood);
        
    end

    % function return value (loglikelihood, state_variable)
    estimatedStates = zeros(2, iterations);
    for t = 1 : iterations
        states_t = stateList{t};
        weights_t = weightList{t};
        alpha_t = states_t(1, :) * weights_t';
        beta_t = states_t(2, :) * weights_t';
        estimatedStates(1, t) = alpha_t;
        estimatedStates(2, t) = beta_t;
        state_variable = estimatedStates; 
    end 

    loglikelihood = -real(loglikelihood); 

end