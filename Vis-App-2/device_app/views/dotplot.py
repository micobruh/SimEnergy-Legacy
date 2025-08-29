import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
import numpy as np
from ..config import appliances_lst, appliances_power

class Dotplot(html.Div):
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

# If 3d, multiply x-axis and y-axis
# log(x) * log(y)


    def update(self, data_needed):
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

        duration_lst = []
        power_lst = []
        for appliance in appliances_lst:
            # Duration
            min_on_watt = appliances_power[appliance]["on_power_threshold_watt"]
            min_on_time = appliances_power[appliance]["min_on_duration_sec"]
            time_instants = new_df[new_df["pred_" + appliance] >= min_on_watt]['unix'].to_numpy()
            durations = []
            if len(time_instants) == 0:
                duration_lst.append(1)
            else:
                current_duration = [time_instants[0]]

                for i in range(1, len(time_instants)):
                    if time_instants[i] - current_duration[-1] <= 6:
                        current_duration.append(time_instants[i])
                    else:
                        durations.append(current_duration)
                        current_duration = [time_instants[i]]

                durations.append(current_duration)
                valid_duration = np.array([(len(i) - 1) * 6 for i in durations if (len(i) - 1) * 6 >= min_on_time])
                average_duration = np.mean(valid_duration)
                # if average_duration == 0:
                #     average_duration = 1
                duration_lst.append(average_duration)

            # Power
            power_lst.append(np.trapz(new_df["pred_" + appliance], dx = 6))
        
        # log_duration_lst = np.log(duration_lst)
        # log_power_lst = np.log(power_lst)
        # log_duration_power_lst = log_duration_lst * log_power_lst

        # trace_lst = [go.Scatter3d(x = [log_duration_lst[i]], y = [log_power_lst[i]], z = [log_duration_power_lst[i]], 
        #                         mode = 'markers', name = appliances_lst[i].replace("_", " ").title())
        #              for i in range(len(appliances_lst))]
        trace_lst = [go.Scatter(x = [duration_lst[i]], y = [power_lst[i]], 
                                mode = 'markers', name = appliances_lst[i].replace("_", " ").title())
                     for i in range(len(appliances_lst))]        
        for i in trace_lst:
            self.fig.add_trace(i)
        
        # Add the main title of the graph
        self.fig.update_layout(
            title_text = f"Log-Log Graph of On-Duration and Average Power Over the {data_needed}",
            xaxis_title = "Log-Scale Average On-Duration in Second",
            yaxis_title = "Log-Scale Average Power in Watt-Second",
            shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}],
            xaxis_range = [-1, 9], 
            yaxis_range = [-1, 9]
        )

        self.fig.update_xaxes(type = "log")
        self.fig.update_yaxes(type = "log")

        return self.fig        

    def reload_df(self, df):
        self.df = df