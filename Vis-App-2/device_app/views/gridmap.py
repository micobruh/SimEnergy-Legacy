import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
import numpy as np
from ..config import appliances_lst, max_power_lst, past_lst


class GridMap(html.Div):
    def __init__(self, name, df):
        self.html_id = name.lower().replace(" ", "-")
        self.df = df

        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(name),
                dcc.Graph(id = self.html_id)
            ],
        )
    
    def update(self, n, labels = appliances_lst):
        self.fig = go.Figure()

        current_time = self.df["unix"].max()
        time_lst = [60 * 60, 60 * 60 * 24, 60 * 60 * 24 * 7, 60 * 60 * 24 * 30, 60 * 60 * 24 * 365]
        
        def find_color(time_now, total_time):
            time_limit = current_time - time_now
            new_df = self.df.query("unix >= @time_limit & unix <= @current_time")
            max_energy_lst = max_power_lst / 1000
            power_lst = np.array([np.trapz(new_df["pred_" + label], dx = 6) for label in labels])
            color_lst = np.where(power_lst > max_energy_lst, 0, 1)
            return color_lst
        
        color_matrix = [find_color(current_time, time_val) for time_val in time_lst]
        color_matrix = [[0,1,0,0,0,0,1,1,1,0],[1,0,1,1,0,0,0,1,1,0],[1,1,0,0,0,0,1,0,1,0],[1,0,0,0,0,1,0,0,0,1],[1,1,1,0,0,0,1,1,1,0]]
        
        label_lst = [label.replace("_", " ").title() for label in labels]
        self.fig = go.Figure(data = go.Heatmap(z = color_matrix, colorscale = [[0, 'red'], [1, 'green']], showscale = False))
        self.fig.update_layout(title_text = "Power Gridmap", 
                xaxis = {'title': 'Appliances', 'tickvals': list(range(len(label_lst))), 'ticktext': label_lst}, 
                yaxis = {'title': 'Time Past', 'tickvals': list(range(len(past_lst))), 'ticktext': past_lst})
        return self.fig

    def reload_df(self, df):
        self.df = df
