import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from nsetools import Nse


yahoo_ticker=pd.read_csv('Yahoo Ticker Symbols.csv', encoding='latin-1')
yahoo_ticker=yahoo_ticker[yahoo_ticker['Exchange']=='NSI']
yahoo_ticker['Ticker']=yahoo_ticker['Ticker'].apply(lambda x: x[:-3])


# Create the Dash app
app = dash.Dash(__name__)


# Initialize an empty DataFrame to track the portfolio
portfolio_df = pd.DataFrame(columns=['Stock Ticker','Company Name','Shares Held','Buy Price'])

# App layout
app.layout = html.Div([
    html.H2("Portfolio Builder"),
    
    # Company name dropdown with autofill functionality
    dcc.Dropdown(
        id='company-dropdown',
        options=[{'label': row['Name'], 'value': row['Ticker']} for index, row in yahoo_ticker.iterrows()],  # Using company name for options
        placeholder='Enter or select a company name',
        multi=False
    ),
    
    dcc.Input(id='ticker-input', type='text', placeholder='Ticker will autofill'),
    
    # Input fields for shares and total money spent
    dcc.Input(id='shares-input', type='number', placeholder='Number of Shares', min=1),
    dcc.Input(id='total-spent-input', type='number', placeholder='Total Money Spent', min=0.01, step=0.01),

    html.Button('Next', id='next-button', n_clicks=0),
    html.Button('Finish', id='finish-button', n_clicks=0),
    
    # Real-time portfolio table updated dynamically
    html.H3("Current Portfolio"),
    dash_table.DataTable(
        id='portfolio-table',
        columns=[
            {"name": "Ticker", "id": "Stock Ticker"},
            {"name": "Company Name", "id": "Company Name"},  
            {"name": "Shares", "id": "Shares Held"},
            {"name": "Average Price", "id": "Buy Price"}
        ],
        data=[],
        style_table={'width': '60%', 'margin': 'auto'}
    ),
    
    html.Div(id='output-message', children='Enter the stock details and press "Next" to continue or "Finish" to save.')
])


@app.callback(
    Output('ticker-input', 'value'),
    [Input('company-dropdown', 'value')]
)
def update_ticker(company_ticker):
    return company_ticker if company_ticker else ''

@app.callback(
    [Output('portfolio-table', 'data'),
     Output('output-message', 'children')],
    [Input('next-button', 'n_clicks'),
     Input('finish-button', 'n_clicks')],
    [State('company-dropdown', 'value'),  
     State('ticker-input', 'value'),
     State('shares-input', 'value'),
     State('total-spent-input', 'value')]
)
def update_or_finish(next_clicks, finish_clicks, company, ticker, shares, total_spent):
    """Main Function where the portfolio is updated"""
    global portfolio_df
    
    ctx = dash.callback_context
    triggered_button = ctx.triggered[0]['prop_id'].split('.')[0]
    
    
    if triggered_button == 'next-button' and next_clicks > 0:
        # Validation of entered inputs
        if not ticker:
            return portfolio_df.to_dict('records'), "Error: Stock ticker cannot be empty."
        if shares <= 0:
            return portfolio_df.to_dict('records'), "Error: Number of shares must be greater than zero."
        if total_spent <= 0:
            return portfolio_df.to_dict('records'), "Error: Total money spent must be greater than zero."

        # Calculate average price
        average_price = total_spent / shares
        company = next(item['label'] for item in [{'label': row['Name'], 'value': row['Ticker']} for index, row in yahoo_ticker.iterrows()] if item['value'] == ticker)

        # Check if the stock already exists in the portfolio
        if ticker in portfolio_df['Stock Ticker'].values:
            # Update the existing stock entry
            existing_row = portfolio_df[portfolio_df['Stock Ticker'] == ticker]
            total_shares = existing_row['Shares Held'].values[0] + shares
            total_cost = (existing_row['Shares Held'].values[0] * existing_row['Buy Price'].values[0]) + total_spent
            new_average_price = total_cost / total_shares
            portfolio_df.loc[portfolio_df['Stock Ticker'] == ticker, 'Shares Held'] = total_shares
            portfolio_df.loc[portfolio_df['Stock Ticker'] == ticker, 'Buy Price'] = new_average_price
            
            return portfolio_df.to_dict('records'), f"Updated stock {ticker} with {shares} additional shares at an average price of {new_average_price:.2f}."
        else:
            # Add a new stock entry
            new_entry = pd.DataFrame({
                'Stock Ticker': [ticker],
                'Company Name': [company],  
                'Shares Held': [shares],
                'Buy Price': [average_price]
            })
            portfolio_df = pd.concat([portfolio_df, new_entry], ignore_index=True)
            
            return portfolio_df.to_dict('records'), f"Added stock {ticker} with {shares} shares at an average price of {average_price:.2f}."


    elif triggered_button == 'finish-button' and finish_clicks > 0:
        # Save the dataframe to a CSV file 'portfolio.csv'
        portfolio_df.to_csv('portfolio.csv', index=False)
        return portfolio_df.to_dict('records'), "Portfolio saved to portfolio.csv."

    return portfolio_df.to_dict('records'), "Enter stock details."


if __name__ == '__main__':
    app.run_server(debug=True)
