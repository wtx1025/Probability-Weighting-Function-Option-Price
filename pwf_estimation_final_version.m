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

%% run particle filter
%{
numberIterations = 100; 
params = [1.4, 1.4, 0.48, 0.9, 0.02, 0.04, 0.001, 0.17, 3.6];
estimatedStates = particle_filter(params, data);

figure;
subplot(2, 1, 1);
plot(1:numberIterations, estimatedStates(1, :), 'b-', 'LineWidth', 1.5);
title('Estimated Alpha Dynamics (Unoptimized)');
xlabel('Iteration');
ylabel('Alpha');
grid on;

subplot(2, 1, 2);
plot(1:numberIterations, estimatedStates(2, :), 'r-', 'LineWidth', 1.5);
title('Estimated Beta Dynamics (Unoptimized)');
xlabel('Iteration');
ylabel('Beta');
grid on;
%}

%% Parameter estimation
initialParams = [1, 1, 0.8, 0.8, 0.1, 0.1, 0.01, 0.1, 2];

lb = [0.4, 0.6, 0.5, 0.5, 0.01, 0.01, 0.001, 0.01, 1];
ub = [1.6, 1.4, 1, 1, 0.15, 0.15, 0.02, 0.2, 6]; 

options = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'Display', 'iter', ...
    'MaxIterations', 100); 

objFunc = @(params) particle_filter(params, data);
[optimalParams, optimalFval] = fmincon(objFunc, initialParams,...
    [], [], [], [], lb, ub, [], options);

fprintf('Optimal Parameters: %s\n', mat2str(optimalParams));
fprintf('Function minimum value: f(x) = %.4f\n', optimalFval);

