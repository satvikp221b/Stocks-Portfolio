import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import timedelta
from sklearn.linear_model import LinearRegression


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Example risk-free rate (annualized 3% divided by 252 trading days)
RISK_FREE_RATE = 0.03/252


portfolio_df = pd.read_csv('data/portfolio.csv',index_col=0)
portfolio_df.dropna(subset=['Stock Ticker'],inplace=True)


def fetch_market_data():
    """Fetch NIFTY 50 data for market comparison"""
    market_data = yf.download('^NSEI',period='1y',interval='1d')
    market_data['Daily Return'] = market_data['Close'].pct_change()
    return market_data['Daily Return'].dropna()


def fetch_pe_ratio(stock_ticker):
    """"Fetch the P/E ratio for a given stock"""
    stock_info = yf.Ticker(stock_ticker).info
    eps = stock_info.get('trailingEps', None)  
    current_price = stock_info.get('currentPrice', None)
    
    # Calculate P/E ratio if EPS is available
    if eps is not None and current_price is not None and eps > 0:
        pe_ratio = current_price / eps
    else:
        pe_ratio = None  
    
    return pe_ratio

def fetch_dividend_yield(stock_ticker):
    """Fetch the Dividend Yield for a given stock"""
    stock_info = yf.Ticker(stock_ticker).info
    dividend_yield = stock_info.get('dividendYield', None)
    
    # Return dividend yield as a percentage
    if dividend_yield is not None:
        dividend_yield *= 100
    return dividend_yield

def fetch_pb_ratio(stock_ticker):
    """Fetch the P/B ratio for a given stock"""
    stock_info = yf.Ticker(stock_ticker).info
    book_value = stock_info.get('bookValue', None)
    current_price = stock_info.get('currentPrice', None)
    
    # Calculate P/B ratio if book value is available
    if book_value is not None and current_price is not None and book_value > 0:
        pb_ratio = current_price/book_value
    else:
        pb_ratio = None
    
    return pb_ratio

def fetch_roe(stock_ticker):
    """Fetch the return on equity for a given stock"""
    stock_info = yf.Ticker(stock_ticker).info
    roe = stock_info.get('returnOnEquity', None)
    
    # Return ROE as a percentage if available
    if roe is not None:
        roe *= 100
    return roe

def fetch_analyst_recommendation(stock_ticker):
    """Fetch the yahoo analyst recommendation for a given stock"""
    stock_info = yf.Ticker(stock_ticker).info
    recommendation_mean = stock_info.get('recommendationMean',None)
    return recommendation_mean

def fetch_market_cap(stock_ticker):
    """Fetch the Market Cap for a given stock"""
    stock_info=yf.Ticker(stock_ticker).info
    market_cap=stock_info.get('marketCap',None)
    return market_cap

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.03/252):
    """Calculate the Sharpe Ratio for the portfolio"""
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()

    # Calculate Sharpe Ratio
    sharpe_ratio = (mean_return-risk_free_rate)/std_dev if std_dev!= 0 else None
    return sharpe_ratio

def format_number(num):
    """Format the number to turn into Trillions, Billions, Millions and thousands"""
    if num >= 1_000_000_000_000:
        return f"{num/1_000_000_000_000:.2f} T"
    elif num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f} B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f} M"
    elif num >= 1_000:
        return f"{num/1_000:.2f} K"
    else:
        return f"{num:.2f}"


def calculate_portfolio_stats_with_market(portfolio_df):
    """Main function to calculate portfolio and stock level based statistics"""
    stock_stats = []
    total_investment = 0
    total_current_value = 0
    portfolio_returns = []
    
    
    market_returns = fetch_market_data()

    for index, row in portfolio_df.iterrows():
        symbol = row['Stock Ticker']
        shares_held = row['Shares Held']
        buy_price = row['Buy Price']
        # Fetch stock data
        stock_data = yf.download(symbol+'.NS', period='1y', interval='1d')
        print(symbol+'.NS')
        pe_ratio = fetch_pe_ratio(symbol+'.NS')  
        dividend_yield = fetch_dividend_yield(symbol+'.NS')  
        pb_ratio=fetch_pb_ratio(symbol+'.NS')
        roe=fetch_roe(symbol+'.NS')
        analyst=fetch_analyst_recommendation(symbol+'.NS')
        market_cap=fetch_market_cap(symbol+'.NS')
        market_cap=format_number(market_cap)
        if not stock_data.empty:
            current_price = stock_data['Close'].iloc[-1]
            previous_price = stock_data['Close'].iloc[-2]

            stock_data['Daily Return'] = stock_data['Close'].pct_change()
            portfolio_returns.append(stock_data['Daily Return'].dropna())
        else:
            current_price = 0

        investment_value = shares_held * buy_price
        current_value = shares_held * current_price
        profit_loss = current_value - investment_value

        total_investment += investment_value
        total_current_value += current_value
        profit_loss_percentage = ((current_price - buy_price) / buy_price) * 100  # Profit/Loss %
        daily_profit_loss = (current_price - previous_price) * shares_held  # Daily profit/loss
        daily_profit_loss_percentage = ((current_price - previous_price) / previous_price) * 100  # Daily profit/loss %        
        print(symbol,shares_held,buy_price)
        stock_stats.append({
            'Stock Ticker': symbol,
            'Shares Held': shares_held,
            'Buy Price':buy_price,
            'Current Price': current_price,
            'Investment Value': investment_value,
            'Current Value': current_value,
            'Profit/Loss': profit_loss,
            'Profit/Loss Percentage':profit_loss_percentage,
            'Daily Profit/Loss':daily_profit_loss,
            'Daily Profit/Loss Percentage':daily_profit_loss_percentage,
            'Market Cap':market_cap,
            'P/E Ratio': pe_ratio,
            'P/B Ratio':pb_ratio,
            'Dividend Yield': dividend_yield,
            'ROE':roe,
            'Analyst Recommendation':analyst
        })

    total_profit_loss = total_current_value - total_investment
    
    # Combine all daily returns from individual stocks
    if portfolio_returns:
        portfolio_returns = pd.concat(portfolio_returns, axis=1).mean(axis=1)
        portfolio_mean_return = portfolio_returns.mean()
        portfolio_std_dev = portfolio_returns.std()
        sharpe_ratio=calculate_sharpe_ratio(portfolio_returns)
        
        
        #Align their indexes
        portfolio_returns, market_returns = portfolio_returns.align(market_returns, join='inner')

        # Calculate Beta using linear regression
        if len(portfolio_returns) == len(market_returns):  
            
            model = LinearRegression()
            model.fit(market_returns.values.reshape(-1, 1), portfolio_returns)
            beta = model.coef_[0]
        else:
            beta = None
            
        # Calculate Treynor Ratio
        treynor_ratio = (portfolio_mean_return - RISK_FREE_RATE) / beta if beta and beta != 0 else 0
        
        
        # Calculate Jensen's Alpha
        market_mean_return = market_returns.mean()
        alpha = (portfolio_mean_return - RISK_FREE_RATE) - (beta * (market_mean_return - RISK_FREE_RATE)) if beta is not None else 0
    
        # Calculate Volatility
        volatility = portfolio_std_dev  

    else:
        beta = None
        treynor_ratio = None
        alpha = None
        volatility = None
    total_investment=format_number(total_investment)
    total_current_value=format_number(total_current_value)
    total_profit_loss=format_number(total_profit_loss)    
    portfolio_summary = {
        'Total Investment': total_investment,
        'Total Current Value': total_current_value,
        'Total Profit/Loss': total_profit_loss,
        'Sharpe Ratio':sharpe_ratio,
        'Beta': beta,
        'Treynor Ratio': treynor_ratio,
        'Jensen Alpha': alpha,
        'Volatility': volatility
    }

    return stock_stats, portfolio_summary

def calculate_sma(stock_data, window):
    """Calculate Simple Moving Average (SMA) over a specified window"""
    stock_data['SMA'] = stock_data['Close'].rolling(window=int(window)).mean()
    return stock_data

def calculate_goldencross(stock_data):
    """Calculate the SMA over two specific window, one 7 day one and one 60 day one"""
    stock_data['SMA Low']=stock_data['Close'].rolling(window=int(7)).mean()
    stock_data['SMA High']=stock_data['Close'].rolling(window=int(60)).mean()
    return stock_data

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels based on the high and low price"""
    diff = high - low
    levels = {
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
        "100%": low
    }
    return levels

def calculate_bollinger_bands(stock_data, window=20):
    """Bollinger Bands"""
    sma = stock_data['Close'].rolling(window=window).mean()
    std_dev = stock_data['Close'].rolling(window=window).std()
    stock_data['Upper Band'] = sma + (std_dev * 2)
    stock_data['Lower Band'] = sma - (std_dev * 2)
    return stock_data

def calculate_vwma(stock_data, window=30):
    """Calculate the Volume-Weighted Moving Average (VWMA) for the stock data"""
    # Calculate VWMA
    price_volume_product = stock_data['Close'] * stock_data['Volume']
    vwma = price_volume_product.rolling(window=window).sum() / stock_data['Volume'].rolling(window=window).sum()
    
    # Add the VWMA to the stock data DataFrame
    stock_data[f'VWMA {window}'] = vwma
    return stock_data

def plot_stock_graph(stock_ticker, overlay_options=None, sma_option=None, sma_window=30, display_period_days=30):
    """Plot the stock's close price and optionally add SMA, Volume, Bollinger Bands, Candlestick based on selected checklist options."""
    # Download a larger dataset for calculating SMA and Bollinger Bands (1 year of data)
    full_stock_data = yf.download(stock_ticker + ".NS", period='1y', interval='1d')

    if sma_option and 'sma' in sma_option and sma_window is not None:
        full_stock_data = calculate_sma(full_stock_data, sma_window)
    
    if overlay_options and 'bollinger' in overlay_options:
        full_stock_data = calculate_bollinger_bands(full_stock_data, sma_window)
    
    if overlay_options and 'goldencross' in overlay_options:
        full_stock_data = calculate_goldencross(full_stock_data)
        
    if overlay_options and 'vwma' in overlay_options:
        full_stock_data = calculate_vwma(full_stock_data,sma_window)
    # Filter the data to display only a portion of the graph depending on the selected time period
    end_date = full_stock_data.index[-1]
    start_date = end_date - timedelta(days=display_period_days)
    display_stock_data = full_stock_data.loc[start_date:end_date]
    
    # Create a figure
    fig = go.Figure()

    # Add candlestick if selected
    if overlay_options and 'candlestick' in overlay_options:
        fig.add_trace(go.Candlestick(
            x=display_stock_data.index,
            open=display_stock_data['Open'],
            high=display_stock_data['High'],
            low=display_stock_data['Low'],
            close=display_stock_data['Close'],
            name='Candlestick'
        ))
    else:
        # Plot the stock's close price if candlestick is not selected
        fig.add_trace(go.Scatter(x=display_stock_data.index, y=display_stock_data['Close'], mode='lines', name='Close Price'))

    # Add SMA if selected
    if sma_option and 'sma' in sma_option:
        fig.add_trace(go.Scatter(x=display_stock_data.index, y=display_stock_data['SMA'], mode='lines', name=f'{sma_window}-Day SMA',line=dict(dash='dash')))
    
    if overlay_options and 'vwma' in overlay_options:
        fig.add_trace(go.Scatter(x=display_stock_data.index, y=display_stock_data[f'VWMA {sma_window}'], mode='lines', name=f'{sma_window}-Day VWMA', line=dict(dash='dash')))


    # Add Volume if selected
    if overlay_options and 'volume' in overlay_options:
        fig.add_trace(go.Bar(x=display_stock_data.index, y=display_stock_data['Volume'], name='Volume', yaxis='y2', marker_color='rgba(255, 0, 0, 0.5)'))

    # Add Bollinger Bands if selected
    if overlay_options and 'bollinger' in overlay_options:
        fig.add_trace(go.Scatter(x=display_stock_data.index, y=display_stock_data['Upper Band'], mode='lines', name='Upper Band', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=display_stock_data.index, y=display_stock_data['Lower Band'], mode='lines', name='Lower Band', line=dict(dash='dash')))
    
    # Add Golden Cross and Death Cross if selected
    if overlay_options and 'goldencross' in overlay_options:
        fig.add_trace(go.Scatter(x=display_stock_data.index, y=display_stock_data['SMA Low'], mode='lines', name='7 - Day SMA',line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=display_stock_data.index, y=display_stock_data['SMA High'], mode='lines', name='60 - Day SMA',line=dict(dash='dash')))

        # Calculate Golden Cross and Death Cross points with linear interpolation
        golden_crosses = []
        death_crosses = []

        for i in range(1, len(display_stock_data)):
            sma_low_prev = display_stock_data['SMA Low'].iloc[i - 1]
            sma_high_prev = display_stock_data['SMA High'].iloc[i - 1]
            sma_low_curr = display_stock_data['SMA Low'].iloc[i]
            sma_high_curr = display_stock_data['SMA High'].iloc[i]

            # Check for Golden Cross (short-term SMA crossing above long-term SMA)
            if sma_low_prev < sma_high_prev and sma_low_curr > sma_high_curr:
                # Linear interpolation
                time_delta = (display_stock_data.index[i] - display_stock_data.index[i - 1]).total_seconds()
                sma_diff_prev = sma_high_prev - sma_low_prev
                sma_diff_curr = sma_low_curr - sma_high_curr
                cross_ratio = sma_diff_prev / (sma_diff_prev + sma_diff_curr)
                cross_x = display_stock_data.index[i - 1] + pd.Timedelta(seconds=cross_ratio * time_delta)
                cross_y = sma_low_prev + cross_ratio * (sma_low_curr - sma_low_prev)
                golden_crosses.append((cross_x, cross_y))

            # Check for Death Cross (short-term SMA crossing below long-term SMA)
            if sma_low_prev > sma_high_prev and sma_low_curr < sma_high_curr:
                # Linear interpolation
                time_delta = (display_stock_data.index[i] - display_stock_data.index[i - 1]).total_seconds()
                sma_diff_prev = sma_low_prev - sma_high_prev
                sma_diff_curr = sma_high_curr - sma_low_curr
                cross_ratio = sma_diff_prev / (sma_diff_prev + sma_diff_curr)
                cross_x = display_stock_data.index[i - 1] + pd.Timedelta(seconds=cross_ratio * time_delta)
                cross_y = sma_low_prev + cross_ratio * (sma_low_curr - sma_low_prev)
                death_crosses.append((cross_x, cross_y))

        # Plot Golden Cross points
        if golden_crosses:
            fig.add_trace(go.Scatter(
                x=[cross[0] for cross in golden_crosses],
                y=[cross[1] for cross in golden_crosses],
                mode='markers',
                marker=dict(color='gold', size=10, symbol='star'),
                name='Golden Cross'
            ))

        # Plot Death Cross points
        if death_crosses:
            fig.add_trace(go.Scatter(
                x=[cross[0] for cross in death_crosses],
                y=[cross[1] for cross in death_crosses],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Death Cross'
            ))

    if overlay_options and 'volume' in overlay_options:
        fig.update_layout(
            yaxis2=dict(
                overlaying='y',
                side='right',
                showgrid=False,
                title='Volume'
            )
        )

    fig.update_layout(
        title=f"{stock_ticker} Stock Price with Selected Overlays",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    return fig  

    
    return fig, None  

def stock_details_layout():
    """Stock Details Layout"""
    return dbc.Container([
        html.H3("Stock Details"),
        dcc.Dropdown(
            id="stock_dropdown",
            options=[{'label': ticker, 'value': ticker} for ticker in portfolio_df['Stock Ticker']],
            placeholder="Select a stock"
        ),
        dcc.Dropdown(
            id="time_period_dropdown",
            options=[
                {'label': '1 Month', 'value': '1mo'},
                {'label': '3 Months', 'value': '3mo'},
                {'label':'6 Months','value':'6mo'},
                {'label': '1 Year', 'value': '1y'},               
            ],
            placeholder="Select time period",
            style={'margin-top': '10px'}
        ),
        html.Div([
            dcc.Checklist(
                id="overlay_option",
                options=[
                    {'label': 'Volume ', 'value': 'volume'},
                    {'label': 'Bollinger Bands ', 'value': 'bollinger'},
                    {'label': 'Candlestick ', 'value': 'candlestick'},
                    {'label':'Golden Cross SMA', 'value': 'goldencross'},
                    {'label': 'Volume Weighted MA', 'value': 'vwma'}
                ],
                style={'margin-top': '10px'},
                inline=True
            ),
            dcc.Checklist(
                id="sma_option",
                options=[{'label': 'Show SMA', 'value': 'sma'}],
                style={'margin-top': '10px'},
                inline=True
            ),
            dcc.Input(
                id="sma_input", 
                type="number", 
                placeholder="Enter SMA window (e.g., 15, 30, 60)", 
                min=1, 
                style={'margin-top': '10px', 'width': '100%'}
            )
        ]),
        dcc.Graph(id="stock_graph"),
        dcc.Graph(id="rsi_graph", style={'display': 'none'})  # Initially hide the RSI graph
    ])



def portfolio_overview_layout():
    """Define the portfolio layout"""
    # Calculate portfolio stats
    stock_stats, portfolio_summary = calculate_portfolio_stats_with_market(portfolio_df)

    # Create stockwise stats table 
    stockwise_stats_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Stock Ticker"), 
            html.Th("Shares Held"), 
            html.Th("Buy Price"),
            html.Th("Current Price"),
            html.Th("Investment Value"), 
            html.Th("Current Value"), 
            html.Th("Profit/Loss"), 
            html.Th("Profit/Loss (%)"),
            html.Th("Daily Profit Loss"),
            html.Th("Daily Profit Loss (%)"),
            html.Th("Market Cap (₹)"),
            html.Th("P/E Ratio"), 
            html.Th("P/B Ratio"),
            html.Th("Dividend Yield"),
            html.Th("ROE"),
            html.Th("Analyst Recommendation")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(stock['Stock Ticker']), 
                html.Td(stock['Shares Held']), 
                html.Td(f"{stock['Buy Price']:.2f}" if stock['Buy Price'] is not None else "N/A"),
                html.Td(f"{stock['Current Price']:.2f}" if stock['Current Price'] is not None else "N/A"),                
                html.Td(f"{stock['Investment Value']:.2f}" if stock['Investment Value'] is not None else "N/A"), 
                html.Td(f"{stock['Current Value']:.2f}" if stock['Current Value'] is not None else "N/A"), 
                html.Td(f"{stock['Profit/Loss']:.2f}" if stock['Profit/Loss'] is not None else "N/A"),
                html.Td(f"{stock['Profit/Loss Percentage']:.2f}" if stock['Profit/Loss Percentage'] is not None else "N/A"),
                html.Td(f"{stock['Daily Profit/Loss']:.2f}" if stock['Daily Profit/Loss'] is not None else "N/A"),
                html.Td(f"{stock['Daily Profit/Loss Percentage']:.2f}" if stock['Daily Profit/Loss Percentage'] is not None else "N/A"),
                html.Td(f"{stock['Market Cap']}" if stock['Market Cap'] is not None else "N/A"),
                html.Td(f"{stock['P/E Ratio']:.2f}" if stock['P/E Ratio'] is not None else "N/A"),
                html.Td(f"{stock['P/B Ratio']:.2f}" if stock['P/B Ratio'] is not None else "N/A"),
                html.Td(f"{stock['Dividend Yield']:.2f}%" if stock['Dividend Yield'] is not None else "N/A"),
                html.Td(f"{stock['ROE']:.2f}%" if stock['ROE'] is not None else "N/A"),
                html.Td(f"{stock['Analyst Recommendation']:.2f}" if stock['Analyst Recommendation'] is not None else "N/A")
            ])
            for stock in stock_stats  
        ])
    ], bordered=True, hover=True, striped=True)

    portfolio_summary_stats = html.Div([
        html.P(f"Total Investment: ₹{portfolio_summary['Total Investment']}"),
        html.P(f"Total Current Value: ₹{portfolio_summary['Total Current Value']}"),
        html.P(f"Total Profit/Loss: ₹{portfolio_summary['Total Profit/Loss']}"),
        html.P(f"Sharpe Ratio: {portfolio_summary['Sharpe Ratio']:.2f}" if portfolio_summary['Sharpe Ratio'] is not None else "Sharpe Ratio: N/A"),
        html.P(f"Beta: {portfolio_summary['Beta']:.2f}" if portfolio_summary['Beta'] is not None else "Beta: N/A"),
        html.P(f"Treynor Ratio: {portfolio_summary['Treynor Ratio']:.5f}" if portfolio_summary['Treynor Ratio'] is not None else "Treynor Ratio: N/A"),
        html.P(f"Jensen's Alpha: {portfolio_summary['Jensen Alpha']:.5f}" if portfolio_summary['Jensen Alpha'] is not None else "Jensen's Alpha: N/A"),
        html.P(f"Volatility: {portfolio_summary['Volatility']:.2%}" if portfolio_summary['Volatility'] is not None else "Volatility: N/A"),
        
    ])

    # Combine everything in the layout
    return dbc.Container([
        html.H3("Overall Portfolio Stats"),
        portfolio_summary_stats,
        html.Hr(),  
        html.H3("Stockwise Stats"),
        stockwise_stats_table
    ])

# Layout of the dashboard
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label="Portfolio Overview", tab_id="portfolio_tab"),
        dbc.Tab(label="Stock Details", tab_id="stock_tab")
    ], id="tabs", active_tab="portfolio_tab"),
    
    html.Div(id="tab_content")
])

# Callback to switch between tabs
@app.callback(Output("tab_content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    # Portfolio Details
    if active_tab == "portfolio_tab":
        return portfolio_overview_layout()
    # Stock Details
    if active_tab == "stock_tab":
        return stock_details_layout()


@app.callback(
    Output("stock_graph", "figure"),  
    [Input("stock_dropdown", "value"), 
     Input("time_period_dropdown", "value"), 
     Input("overlay_option", "value"), 
     Input("sma_option", "value"), 
     Input("sma_input", "value")],
    prevent_initial_call=True
)
def update_stock_graph(stock_ticker, time_period, overlay_options, sma_option, sma_window):
    """Main Function for stock level chart update """
    # If no stock or time period is selected, return an empty figure
    if stock_ticker is None or time_period is None:
        return go.Figure()  

    # Convert the selected time period into the number of days
    time_period_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
    display_period_days = time_period_map.get(time_period, 30)


    fig = plot_stock_graph(stock_ticker, overlay_options, sma_option, sma_window, display_period_days)

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
