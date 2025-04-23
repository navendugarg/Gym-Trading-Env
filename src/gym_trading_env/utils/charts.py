import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def charts(df, lines=[]):
    # Prepare data
    df['date_str'] = df.index.strftime("%Y-%m-%d %H:%M")
    df["cumulative_rewards"] = df["reward"].cumsum()
    
    # Define subplot architecture with row numbers and relative heights
    architecture = {
        "candlesticks": {"row": 1, "height": 0.35},
        "volumes": {"row": 2, "height": 0.09},
        "portfolios": {"row": 3, "height": 0.09},
        "positions": {"row": 4, "height": 0.09},
        "rewards": {"row": 5, "height": 0.09},
    }
    
    # Number of rows for subplots
    num_rows = len(architecture)
    
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[arch["height"] for arch in architecture.values()],
        subplot_titles=("Candlesticks", "Volume", "Portfolio Value", "Positions", "Cumulative Rewards")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date_str'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#06AF8F',
            decreasing_line_color='#FC4242',
            name='Candlesticks'
        ),
        row=architecture["candlesticks"]["row"], col=1
    )
    
    # Add custom lines to candlestick chart
    for line in lines:
        y_data = line["function"](df)
        line_options = line.get("line_options", {})
        fig.add_trace(
            go.Scatter(
                x=df['date_str'],
                y=y_data,
                mode='lines',
                name=line["name"],
                line=dict(color=line_options.get("color", "blue"), width=line_options.get("width", 1))
            ),
            row=architecture["candlesticks"]["row"], col=1
        )
    
    # Volume bar chart
    fig.add_trace(
        go.Bar(
            x=df['date_str'],
            y=df['volume'],
            name='Volume',
            marker_color='blue',
            opacity=0.3
        ),
        row=architecture["volumes"]["row"], col=1
    )
    
    # Portfolio valuation line chart
    fig.add_trace(
        go.Scatter(
            x=df['date_str'],
            y=df['portfolio_valuation'],
            mode='lines',
            name='Portfolio Valuation',
            line=dict(color='blue')
        ),
        row=architecture["portfolios"]["row"], col=1
    )
    
    # Positions step line chart
    fig.add_trace(
        go.Scatter(
            x=df['date_str'],
            y=df['position'],
            mode='lines',
            name='Positions',
            line=dict(color='blue'),
            step='pre'  # Creates a step line
        ),
        row=architecture["positions"]["row"], col=1
    )
    
    # Cumulative rewards area chart
    fig.add_trace(
        go.Scatter(
            x=df['date_str'],
            y=df['cumulative_rewards'],
            mode='lines',
            name='Cumulative Rewards',
            line=dict(color='blue'),
            fill='tozeroy'  # Creates an area chart
        ),
        row=architecture["rewards"]["row"], col=1
    )
    
    # Update layout
    fig.update_layout(
        height=650,
        width=800,
        title_text="Financial Charts",
        showlegend=False,
        xaxis_rangeslider_visible=False,  # Hides the default Plotly range slider
        template='plotly_white'
    )
    
    # Hide x-axis labels for all but the bottom subplot
    for i in range(1, num_rows):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    fig.update_xaxes(showticklabels=True, row=num_rows, col=1)
    
    # Add y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    fig.update_yaxes(title_text="Position", row=4, col=1)
    fig.update_yaxes(title_text="Rewards", row=5, col=1)
    
    return fig