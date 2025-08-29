from dash import Dash
from flask import Flask

server = Flask(__name__)
app = Dash(__name__, server = server, url_base_pathname = '/visualization/')
app.title = "Visualization of Power Consumption"
