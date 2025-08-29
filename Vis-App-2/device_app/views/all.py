from dash import html
import numpy as np

class All(html.Div):
    def __init__(self, name, df):
        self.html_id = name.lower().replace(" ", "-")
        self.df = df

        # Equivalent to `html.Div([...])`
        super().__init__(
            children = [
                html.Div(id = self.html_id)
            ],
        )

    def update(self, data_needed):
        current_time = self.df["unix"].max()
        time_limit = 0
        self.children = []
        
        if data_needed == "Past Hour":
            time_limit = current_time - 60 * 60
        elif data_needed == "Past Day":
            time_limit = current_time - 60 * 60 * 24
        elif data_needed == "Past Week":
            time_limit = current_time - 60 * 60 * 24 * 7
        elif data_needed == "Past Month":
            time_limit = current_time - 60 * 60 * 24 * 30
        elif data_needed == "Past Year":
            time_limit = current_time - 60 * 60 * 24 * 365
        new_df = self.df.query("unix >= @time_limit & unix <= @current_time")
        
        total_power = np.trapz(new_df["mains"], dx = 6) / 1000 / 3600
        electric_price = total_power * 2

        self.children = [html.Div([
            html.H2("Total Power"),
            html.H3(f"Power consumed: {np.round(total_power, 2)} kWh"),
            html.H3(f"Electric Price: €{np.round(electric_price, 2)}"),
            html.Br(),
            ]), 
            html.Div([
            html.H2("Total Gas"),
            html.H3("Gas consumed: 60 kWh"),
            html.H3("Electric Price: €5"),
            html.Br(),
            ]),
            html.Div([
            html.H2("Total Water"),
            html.H3("Water consumed: 500 L"),
            html.H3("Water Price: €10"),
            html.Br(), 
            ])
            ]
        return self.children

    def reload_df(self, df):
        self.df = df
 