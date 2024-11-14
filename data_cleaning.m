%% data cleaning 
% this data includes both call and put that are 0-90 days from expiration 
% data period is from 2000-01-01 to 2023-08-31
% https://help.streetsmart.schwab.com/edge/1.24/Content/Index%20Symbols.htm
% https://www.cheddarflow.com/blog/spx-vs-spxw-a-comprehensive-guide-to-their-differences/

data = readtable("C:\Users\王亭烜\Desktop\Thesis\Data\new data\atm_data.csv");
call_data = data(strcmp(data.cp_flag, 'C'),:);
call_data = call_data(:, {'date','symbol','exdate','cp_flag','strike_price','delta',...
    'best_bid','best_offer','impl_volatility','volume','open_interest'});

call_data = call_data(call_data.date>datetime(2010, 1, 1), :);
call_data = call_data(weekday(call_data.date)==4, :); %filter out the data on Wednesday
% call_data = call_data(contains(call_data.symbol, 'SPX') & ~contains(call_data.symbol,...
%   'SPXW'),:);

%% select data with days-to-expiration closest to 30
target_days = [30]; % we can extend the array if we want to select data with more than one days-to-expiration (ex:30,45,60)
unique_dates = unique(call_data.date);
select_data = table();
call_data.days_to_expiration = days(call_data.exdate-call_data.date);

for i = 1:length(unique_dates)
    current_date = unique_dates(i);
    date_group = call_data(call_data.date==current_date, :);
    for j = 1:length(target_days)
        target = target_days(j);
        min_diff = min(abs(date_group.days_to_expiration - target));
        closest_indices = find(abs(date_group.days_to_expiration - target)==min_diff);
        closest_rows = date_group(closest_indices, :);
        select_data = [select_data; closest_rows];
    end
end 

%% Discard options with zero trading volume or open interest
select_data = select_data(select_data.volume~=0 & select_data.open_interest~=0, :);

%% Implied volatility must be greater than 5% 
select_data = select_data(select_data.impl_volatility > 0.05, :); 

%% Include options with delta between 0.4 and 0.6 
select_data = select_data(select_data.delta >= 0.4 & select_data.delta <= 0.6, :);

%% Mid quotes must be greater than $3/8 
select_data.mid_quotes = (select_data.best_bid + select_data.best_offer) / 2; 
select_data = select_data(select_data.mid_quotes>3/8, :);

%% Left each date with only single option data that has delta closest to 0.5 
final_data = table();
unique_dates = unique(select_data.date);

for i = 1:length(unique_dates)
    current_date = unique_dates(i);
    date_group = select_data(select_data.date == current_date, :);

    [~, closest_index] = min(abs(date_group.delta - 0.5));
    closest_row = date_group(closest_index, :);
    final_data = [final_data; closest_row]; 
end 

%% Summary of the final data 
delta_column = final_data.delta; 
mid_quotes_column = final_data.mid_quotes;
implied_vol_column = final_data.impl_volatility; 
q25 = quantile(delta_column, 0.25);  
q50 = quantile(delta_column, 0.50);  
q75 = quantile(delta_column, 0.75);  
min_val = min(delta_column); 
max_val = max(delta_column); 

num_records = zeros(4,1);
avg_mid_quotes = zeros(4,1);
avg_implied_vol = zeros(4,1);

idx1 = delta_column >= min_val & delta_column < q25;
num_records(1) = sum(idx1);
avg_mid_quotes(1) = mean(mid_quotes_column(idx1));
avg_implied_vol(1) = mean(implied_vol_column(idx1));

idx2 = delta_column >= q25 & delta_column < q50;
num_records(2) = sum(idx2);
avg_mid_quotes(2) = mean(mid_quotes_column(idx2));
avg_implied_vol(2) = mean(implied_vol_column(idx2));

idx3 = delta_column >= q50 & delta_column < q75;
num_records(3) = sum(idx3);
avg_mid_quotes(3) = mean(mid_quotes_column(idx3));
avg_implied_vol(3) = mean(implied_vol_column(idx3));

idx4 = delta_column >= q75 & delta_column < max_val;
num_records(4) = sum(idx4);
avg_mid_quotes(4) = mean(mid_quotes_column(idx4));
avg_implied_vol(4) = mean(implied_vol_column(idx4));

for i = 1:4
    fprintf('Moneyness %d: observations = %d, average mid quotes = %.4f, average implied volatility = %.4f\n', ...
        i, num_records(i), avg_mid_quotes(i), avg_implied_vol(i));
end

%% Export final data as CSV 
writetable(final_data, 'C:\Users\王亭烜\Desktop\Thesis\Data\final_data.csv');


