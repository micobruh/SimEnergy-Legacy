from dash import html, dcc
from dash.dependencies import Input, Output

from device_app.data import get_data
from device_app.main import server, app
from device_app.views.piechart import Piechart
from device_app.views.show import Show
from device_app.views.all import All
from device_app.views.dotplot import Dotplot
from device_app.views.gridmap import GridMap
from device_app.views.menu import make_menu_layout

from flask import redirect, render_template, request
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

@server.route('/')

def home():
    # if request.method == 'POST':
    #     if request.form.get('action1') == 'vis':
    #         render_dashboard()
    # elif request.method == 'GET':
    #     return render_template('home.html')
    
    return render_template('home.html')

@server.route('/visualization/')
def render_dashboard():
    return redirect('/dash1')

# Create data
df = get_data()

# Instantiate custom views
all = All("all", df)
show = Show("show", df)
piechart_1 = Piechart("piechart-energy", df)
piechart_2 = Piechart("piechart-price", df)
dotplot = Dotplot("dotplot", df)
gridmap = GridMap("gridmap", df)

# Define layout
app.layout = html.Div(
    id = "app-container",
    children = [
        # Left column: Define menu in app
        html.Div(
            id = "left-column",
            className = "three columns",
            children = make_menu_layout(False)
        ),

        # Right column: Define 2 tabs/pages in app
        html.Div(
            id = "right-column",
            className = "nine columns",
            children = [
                dcc.Tabs(
                    id = 'tab-aggregator',
                    value = 'view',
                    children = [
                        dcc.Tab(label = 'All View', value = 'all-view'),
                        dcc.Tab(label = 'Show View', value = 'show-view'),
                        dcc.Tab(label = 'Pie Chart View', value = 'pie-view'),
                        dcc.Tab(label = 'Log Log Graph View', value = 'dot-view'),
                        dcc.Tab(label = 'Grid Map View', value = 'gridmap-view')
                    ]
                ),
                # TODO make it fill the rest of the page
                html.Div(id = 'tabs-content')
            ],
        ),

        dcc.Interval(id = 'interval-component', interval = 10000000, n_intervals = 0),
    ],
)

# Generate show in the 1st page of app
@app.callback(
    Output("all", "children"),
    Input("past-selector-dropdown-1", "value"),
)
def update_all(data_needed):
    return all.update(data_needed)

# Generate show in the 1st page of app
@app.callback(
    Output("show", "children"),
    Input("past-selector-dropdown-2", "value"),
)
def update_show(data_needed):
    return show.update(data_needed)

# Generate piechart-energy in the 2nd page of app
@app.callback(
    Output("piechart-energy", "figure"),
    Input("past-selector-dropdown-3", "value")
)
def update_pie_1(data_needed):
    return piechart_1.update(data_needed, "energy")

# Generate piechart-price in the 3rd page of app
@app.callback(
    Output("piechart-price", "figure"),
    Input("past-selector-dropdown-3", "value")
)
def update_pie_2(data_needed):
    return piechart_2.update(data_needed, "price")

# Generate log log graph in the 4th page of app
@app.callback(
    Output("dotplot", "figure"), 
    Input("past-selector-dropdown-4", "value")
)

def update_dotplot(data_needed):
    return dotplot.update(data_needed)

# Generate grid map in the 5th page of app
@app.callback(
    Output('gridmap', 'figure'), 
    Input('interval-component', 'n_intervals')
)

def update_gridmap(n):
    return gridmap.update(n)


# Generate menu on the left side of the app
@app.callback(
    Output('tabs-content', 'children'),
    Output('left-column', 'children'),
    Input('tab-aggregator', 'value')
)
# Update menu based on the type of chart/view chosen
def update_view(tab):
    if tab == 'all-view':
        return html.Div([all]), make_menu_layout(tab)      
    elif tab == 'show-view':
        return html.Div([show]), make_menu_layout(tab)    
    elif tab == 'pie-view':
        return html.Div([piechart_1, piechart_2]), make_menu_layout(tab)
    elif tab == 'dot-view':
        return html.Div([dotplot]), make_menu_layout(tab)    
    elif tab == 'gridmap-view':
        return html.Div([gridmap]), make_menu_layout(tab)

entire_app = DispatcherMiddleware(server, {
    '/dash1': app.server
})

if __name__ == '__main__':
    # entire_app.run_server(debug = False, dev_tools_ui = False, use_reloader = True, host = "127.0.0.1", port = "8050")
    run_simple('127.0.0.1', 8050, entire_app, use_reloader = True, use_debugger = True)
    
