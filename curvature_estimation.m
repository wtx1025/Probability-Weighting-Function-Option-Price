%% read data (option, index, treasury) and merge them 
option_data = readtable("C:\Users\王亭烜\Desktop\Thesis\Data\new data\final_data.csv");
index_data = readtable("C:\Users\王亭烜\Desktop\Thesis\Data\new data\equity_index_data.csv");
treasury_data = readtable("C:\Users\王亭烜\Desktop\Thesis\Data\new data\DTB3.xlsx");        

option_data.Properties.VariableNames{'date'} = 'Date';
index_data.Properties.VariableNames{'caldt'} = 'Date';
treasury_data.Properties.VariableNames{'DATE'} = 'Date';

option_data.Date = datetime(option_data.Date);
index_data.Date = datetime(index_data.Date);
treasury_data.Date = datetime(treasury_data.Date);

data = innerjoin(option_data, index_data, 'Keys', 'Date');
data = innerjoin(data, treasury_data, 'Keys', 'Date');
selected_columns = {'Date','exdate','strike_price','mid_quotes','days_to_expiration','spindx','DTB3'};
data = data(:, selected_columns);
data.DTB3 = fillmissing(data.DTB3, 'previous');
zero_indices = (data.DTB3 == 0);
data.DTB3(zero_indices) = fillmissing(data.DTB3(zero_indices), 'previous'); 
data.DTB3(data.DTB3 < 0) = abs(data.DTB3(data.DTB3 < 0));
data.DTB3 = data.DTB3 .* (1/100) .* (1/3); 


%% calculate equity return's mean and std, option price's std 
daily_ret_mean = mean(index_data.sprtrn);
daily_ret_std = std(index_data.sprtrn);
monthly_ret_mean = daily_ret_mean * 30;
monthly_ret_std = daily_ret_std * sqrt(30);
weekly_option_var = var(data.mid_quotes);

%% Estimate the parameters and state variables using Particle Filter & EM algorithm 
% https://github.com/AleksandarHaber/Python-Implementation-of-Particle-Filter/blob/main/finalVersion.py

% Define the distributions 
meanProcess = [0; 0];
covarianceProcess = [0.05, 0; 0, 0.05];
meanNoise = 0;
covarianceNoise = 800; 

% Define the state space model 
phi_alpha = 0.9;
phi_beta = 0.9; 
A = [phi_alpha, 0; 0, phi_beta];
x0 = [1.2; 1.02];
gamma = 4;
mu = 0.0104;
sigma = 0.107; 
lowerBound = -0.5;
upperBound = 0.8; 
numGrids = 200; 
gridValues = linspace(lowerBound, upperBound, numGrids); 


% Implementation of particle filter
rng(0);
x0Guess = x0 + [0.01; -0.05]; 
[xGrid, yGrid] = meshgrid(x0Guess(1)-0.4:0.05:x0Guess(1)+0.4, x0Guess(2)-0.2:0.05:x0Guess(2)+0.2);
xVec = xGrid(:);
yVec = yGrid(:); 
states = [xVec, yVec]';
[dim1, numberParticle] = size(states);
weights = (1 / numberParticle) * ones(1, numberParticle); 

numberIterations = 702; %702
stateList = {}; 
stateList{end+1} = states; 
weightList = {}; 
weightList{end+1} = weights; 

count = 0; 
for i = 1:numberIterations
    
    newStates = A * states + mvnrnd(meanProcess, covarianceProcess, numberParticle)';

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

        update_alpha = newStates(1, j);
        update_beta = newStates(2, j);

        numerator = 0;
        denominator = 0; 
        for k = 1:length(gridValues)
            currentValue = gridValues(k); 
            stock_price = data{i, 'spindx'}; 
            option_payoff = max(stock_price*(1+gridValues(k)) - data{i, 'strike_price'}/1000, 0); 

            c1 = 0;
            c2 = 0; 
            if k==1
                c2 = normpdf(gridValues(k), mu, sigma) * 0.0065;
                c1 = 0; 
            else
                for s = 1:k-1
                    c1 = c1 + normpdf(gridValues(s), mu, sigma) * 0.0065;
                end
                for t = 1: k
                    c2 = c2 + normpdf(gridValues(t), mu, sigma) * 0.0065; 
                end 
            end 

            numerator = numerator + option_payoff*(1+gridValues(k))^(-gamma)...
                *(exp(-(-newStates(2,j)*log(c2))^newStates(1,j))...
                - exp(-(-newStates(2,j)*log(c1))^newStates(1,j))); %problem is here!
            denominator = denominator + (1+data{i, 'DTB3'})*(1+gridValues(k))^(-gamma)...
                *(exp(-(-newStates(2,j)*log(c2))^newStates(1,j))...
                - exp(-(-newStates(2,j)*log(c1))^newStates(1,j)));
        end 
        
        meanDis = numerator / denominator;
        distribution0 = mvnpdf(data{i,"mid_quotes"}, meanDis, covarianceNoise);
        newWeights(j) = real(distribution0 * weights(j)); 
        fprintf('i=%d j=%d alpha=%.2f beta=%.2f n=%.4f d=%.4f meanDis=%.4f mid_quotes=%.2f prob=%.4f weight=%.6f\n',...
            i, j, update_alpha, update_beta, numerator, denominator, meanDis, data{i,"mid_quotes"},...
            distribution0, newWeights(j)); 

    end 

    %weightStandardized = max(weightStandardized, 1e-6);
    weightStandardized = real(newWeights / sum(newWeights));

    tmp1 = weightStandardized.^2; 
    Neff = 1 / sum(tmp1); 
    if Neff < (numberParticle / 3)
        resampleStateIndex = randsample(1:numberParticle, numberParticle, true, weightStandardized);
        newStates = newStates(:, resampleStateIndex);
        weightStandardized = (1 / numberParticle) * ones(1, numberParticle); 
    end 

    states = newStates; 
    weights = weightStandardized; 
    stateList{end+1} = states;
    weightList{end+1} = weights; 
    count = count + 1; 
    disp('========================================================================================================') 
end 

%% Show the estimate results 
estimatedStates = zeros(2, numberIterations);

for t = 1:numberIterations
    states_t = stateList{t};
    weights_t = weightList{t};
    
    alpha_t = states_t(1, :) * weights_t';
    beta_t = states_t(2, :) * weights_t';
    
    estimatedStates(1, t) = alpha_t;
    estimatedStates(2, t) = beta_t;
end

figure;
subplot(2, 1, 1);
plot(1:numberIterations, estimatedStates(1, :), 'b-', 'LineWidth', 1.5);
title('Estimated Alpha Dynamics');
xlabel('Iteration');
ylabel('Alpha');
grid on;

subplot(2, 1, 2);
plot(1:numberIterations, estimatedStates(2, :), 'r-', 'LineWidth', 1.5);
title('Estimated Beta Dynamics');
xlabel('Iteration');
ylabel('Beta');
grid on;




































