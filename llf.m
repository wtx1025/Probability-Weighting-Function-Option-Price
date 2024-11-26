function llf_value = llf(mu, sigma, gamma, stateList, weightList)

    % import data
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

    % calculate value of log likelihood  
    lowerBound = -0.5;
    upperBound = 0.8; 
    numGrids = 200; 
    gridValues = linspace(lowerBound, upperBound, numGrids);
    covarianceNoise = 800; 
    iter = length(stateList); % length of stateList is total observation + 1
    likelihood = 1; 

    for i = 2 : iter

        % the following code compute loglikelihood for each observed data i
        numerator = 0;
        denominator = 0;
        states_i = stateList{i};
        weights_i = weightList{i};
        alpha_i = states_i(1, :) * weights_i';
        beta_i = states_i(2, :) * weights_i';

        for k = 1:length(gridValues)
            currentValue = gridValues(k); 
            stock_price = data{i-1, 'spindx'}; 
            option_payoff = max(stock_price*(1+gridValues(k)) - data{i-1, 'strike_price'}/1000, 0); 
    
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
                *(exp(-(-beta_i*log(c2))^alpha_i)...
                - exp(-(-beta_i*log(c1))^alpha_i)); 
            denominator = denominator + (1+data{i-1, 'DTB3'})*(1+gridValues(k))^(-gamma)...
                *(exp(-(-beta_i*log(c2))^alpha_i)...
                - exp(-(-beta_i*log(c1))^alpha_i));
        end  
        meanDis = numerator / denominator;
        distribution0 = mvnpdf(data{i-1,"mid_quotes"}, meanDis, covarianceNoise);
        likelihood = likelihood * distribution0; 

    end 
    
    llf_value = log(likelihood); 

end