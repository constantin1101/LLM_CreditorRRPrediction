import pandas as pd

macros_selected_codes = ['MNFCTRIRSA', 'UEMPLT5', 'MORTGAGE30US', 'CPFF']

macros_selected = ['Manufacturers inventories to sales ratio', 
                    'Number of civilians unemployed for less than 5 weeks',
                    '30 year conventional mortgage rate',
                    '3 month commercial paper minus federal funds rate']

industry_dummies = ['ActIndustryDistress1', 'ActIndustryDistress2', 'Materials', 'Communication Services', 'Consumer Discretionary', 'Industrials', 
'Consumer Staples', 'Financials', 'Energy', 'Health Care', 'Utilities', 'Information Technology', 'Real Estate']

industry_labels = ['ActIndustryDistress1', 'ActIndustryDistress2', 'Materials', 'Communication Services', 'Consumer Discretionary', 'Industrials', 
'Consumer Staples', 'Financials', 'Energy', 'Health Care', 'Utilities', 'Information Technology', 'Real Estate']



seniority_dummies = ['Senior_Secured', 'Senior_Unsecured', 'Senior_Subordinate', 'SubordinateJunior']
seniority_labels = ['Senior secured', 'Senior unsecured', 'Senior subordinated', 'Subordinated \& Junior']

bond_dummies = ['OfferingAmount' , 'DaysToMaturity' , 'BOND_COUPON' , 'Rating' , 'CIQ_CDSavailable' , 'COVENANTS' , 'RR']
bond_labels = ['Offering amount', 'Time to maturity', 'Coupon', 'Rating', 'CDS availability', 'Covenants', 'RR']

news_variables = ['Government', 'Intermediation', 'Securities Markets', 'War', 
                    'Unclassified', 'EPU', 'Government.1', 'Natural Disaster', 'NVIX']
company_dummies = ['EquityValue', 'DefaultBarrier2', 'NetIncomeMargin', 'TotalAssets', 'NumberEmployees'] 
company_labels = ['Equity value', 'Default barrier', 'Net income margin', 'Total assets', 'Number of employees'] 

stock_market = ['S\&P Ret Vol', 'S\&P 500', 'VXDCLS', 'Nasdaq Ret', 'VXNCLS', 'Russell Ret', 'Russell2000Vol1m', 'Wilshire Ret', 'WilshireVol1m']

stock_market_labels = ['S\&P 500 Index return', 'S\&P 500 index', 'CBOE DJIA Volatility Index', 'NASDAQ 100 Index return', 'CBOE NASDAQ 100 Volatility Index', 'Russell 2000 Price Index return', 'Russell 2000 Vol 1m', 'Wilshire US Small-Cap Price Index', 'Wilshire Small Cap Vol']

bond_liquidity = ['AvgDailyVOL','TRADES','amihud_ILLIQ','price_dispersion','roll_2','CorwinSchultz2']

bond_liquidity_labels = ['Volume','Trades','Amihud','Price dispersion','Roll','Corwin Schultz']


#def get_macro_vars_list(data_file : str = 'data/20171009_DataMerged.xlsm'):
#    df_macro_exp = pd.read_excel(data_file, sheet_name='TableMacroVariables')
#    macro_vars = list(df_macro_exp['TOTLL'].values) + ['TOTLL', 'Cororate MTN', 'CIQ_Instrument_ID', 'InstrumentType']
#    return macro_vars