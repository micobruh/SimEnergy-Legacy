# Visualization App on the Traffic Accidents in the UK

## About this app

The visualization app aims to show the bar charts, hexmap and heatmap on the dataset regarding traffic accident in the UK. Several interactions can be achieved in the app, such as selecting the attributes of the interest.

## Requirements needed

1. Visual Studio Code is used.
2. Python 3.8 interpreter is used.
3. There is dash with version 2.0.0 or higher in the python interpreter.
4. There is numpy with version 1.21.2 or higher in the python interpreter.
5. There is pandas with version 1.3.3 or higher in the python interpreter.
6. There is plotly with exactly or close to version 5.5.0 in the python interpreter.

## How to run the app

1. Upzip the folder with all codes.
2. Open Visual Studio Code. Make sure all requirements above are met. If not, install all of the necessary packages into the python interpreter.
3. Click "File" at the top left corner and then click "Open Folder" and select the upzip folder. After that, click "Select Folder" so that all files are opened in VS Code.
3. On the left side of VS Code, an explorer should be found with all file names inside the folder. Click "app.py" and open it.
4. Click the triangle button at the top right corner of the VS code to run the app.
5. In the terminal at the bottom, a sentence with the app website is shown (e.g. Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)). Click the link to open the browser and use the visualization app. Make sure the VS code app is still running in the background.

## Our own implementation on the app

1. To reduce the storage memory and running time of the app, we only select several useful attributes from the original dataset and create a new csv file to store these data.
2. We unify all the unknown labels to the same value instead of allowing them to be different values. This allows better filter of the data when running the app.
3. We convert the numerical values of the items per attribute to the short literal meaning. This allows everyone to understand quickly what every unique values in an attribute means.
4. We create new attributes regarding day of week and hour. This allows more useful interpretation of the data.
5. We only remove rows with unknown values if this is the case for the attribute of interest. (e.g. A row has accident severity as unknown and speed limit as known. If we are interested in accident severity, this row is removed from the visualization. But if we are interested in speed limit, this row is kept.)
6. We allow selecting several attributes from many options to be shown in the graph.
7. Several tabs are used to show different graphs instead of showing all of them in one tab. This allows esier navigation in the app.

## What we get from existing libraries
1. We use plotly and dash to plot the graphs (stacked bar chart, hexmap and heatmap) in an app that allows interaction.
2. We make use of the functions in the libraries to define apperance of the app, such as the menu content and the graph appearance.
3. We are able to show certain data on Open Street Map thanks to the library.
