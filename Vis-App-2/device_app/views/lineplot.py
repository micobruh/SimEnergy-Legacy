import plotly.graph_objects as go
from dash import dcc, html
from ..config import appliances_lst

class Lineplot(html.Div):
    def __init__(self, name, df):
        self.html_id = name.lower().replace(" ", "-")
        self.df = df

        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children = [
                html.H6(name),
                dcc.Graph(id = self.html_id)
            ],
        )

    def update(self, data_needed, labels = appliances_lst):
        self.fig = go.Figure()

        current_time = self.df["unix"].max()
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

        trace_lst = [go.Scatter(x = new_df["unix"], y = new_df["pred_" + label], 
                                mode = 'lines', name = label.replace("_", " ").title())
                     for label in labels]
        for i in trace_lst:
            self.fig.add_trace(i)

        # Add titles of x-axis, y-axis and legend
        self.fig.update_layout(
            xaxis_title = "UNIX Time",
            yaxis_title = "Power in Watt",
        )

        # Add the main title of the graph
        self.fig.update_layout(
            title_text = f"Appliance Power Change Over the {data_needed}",
        )

        return self.fig

    def reload_df(self, df):
        self.df = df
 