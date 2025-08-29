import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
import numpy as np
from ..config import appliances_lst


class Piechart(html.Div):
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
    
    def update(self, data_needed, type):
        self.fig = go.Figure()

        current_time = self.df["unix"].max()
        time_limit = 0
        if data_needed == "Past Hour":
            time_limit =  current_time - 60 * 60
        elif data_needed == "Past Day":
            time_limit = current_time - 60 * 60 * 24
        elif data_needed == "Past Week":
            time_limit = current_time - 60 * 60 * 24 * 7
        elif data_needed == "Past Month":
            time_limit = current_time - 60 * 60 * 24 * 30
        elif data_needed == "Past Year":
            time_limit = current_time - 60 * 60 * 24 * 365
        new_df = self.df.query("unix >= @time_limit & unix <= @current_time")

        power_lst = [np.trapz(new_df["pred_" + appliance], dx = 6) / 1000 / 3600 for appliance in appliances_lst]
        power_lst.append(np.trapz(new_df["mains"], dx = 6) / 1000 / 3600 - sum(power_lst))
        label_lst = [appliance.replace("_", " ").title() for appliance in appliances_lst]
        label_lst.append("Other Consumptions")
        
        if type == "energy":
            self.fig.add_trace(go.Pie(labels = label_lst, values = power_lst, hole = .3, textinfo = 'label+percent',
                                insidetextorientation = 'radial', name = "Energy Consumption"))
            self.fig.update_layout(title_text = f"Total Energy over the {data_needed}: {np.round(sum(power_lst), 3)} kWh")            
        elif type == "price":
            self.fig.add_trace(go.Pie(labels = label_lst, values = np.array(power_lst) * 2, hole = .3, textinfo = 'label+percent',
                             insidetextorientation = 'radial', name = "Energy Cost"))
            self.fig.update_layout(title_text = f"Total Price over the {data_needed}: €{np.round(sum(power_lst) * 2, 3)}")             
        return self.fig

    def reload_df(self, df):
        self.df = df
