from dash import html, get_asset_url
import numpy as np
from ..config import appliances_lst, images_lst

class Show(html.Div):
    def __init__(self, name, df):
        self.html_id = name.lower().replace(" ", "-")
        self.df = df

        # Equivalent to `html.Div([...])`
        super().__init__(
            children = [
                html.Div(id = self.html_id)
            ],
        )

    def update(self, data_needed, labels = appliances_lst, images_lst = images_lst):
        current_time = self.df["unix"].max()
        total_time = 0
        self.children = []
        
        if data_needed == "Past Hour":
            total_time = 60 * 60
        elif data_needed == "Past Day":
            total_time = 60 * 60 * 24
        elif data_needed == "Past Week":
            total_time = 60 * 60 * 24 * 7
        elif data_needed == "Past Month":
            total_time = 60 * 60 * 24 * 30
        elif data_needed == "Past Year":
            total_time = 60 * 60 * 24 * 365
        time_limit = current_time - total_time
        new_df = self.df.query("unix >= @time_limit & unix <= @current_time")
        
        power_lst = np.array([np.trapz(new_df["pred_" + label], dx = 6) / 1000 / 3600 for label in labels])
        total_power = np.trapz(new_df["mains"], dx = 6) / 1000 / 3600
        percentage_lst = power_lst / total_power * 100
        price_lst = power_lst * 2
        label_lst = [label.replace("_", " ").title() for label in labels]

        for label, image, power, price, percentage in zip(label_lst, images_lst, power_lst, price_lst, percentage_lst):
            child = html.Div([
                html.H2(label),
                html.Img(src = get_asset_url(image)),
                html.H3(f"Power consumed: {np.round(power, 2)} kWh"),
                html.H3(f"Electric Price: €{np.round(price, 2)}"),
                html.H3(f"Percentage over Total Power: {np.round(percentage, 2)}%"),
                html.Br(),
            ])
            self.children.append(child)
        return self.children

    def reload_df(self, df):
        self.df = df
 