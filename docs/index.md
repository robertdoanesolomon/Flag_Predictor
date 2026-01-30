# Latest Forecast Plots

## Isis
![Isis forecast](spaghetti_rain_isis.png)

## Godstow
![Godstow forecast](spaghetti_rain_godstow.png)

## Wallingford
![Wallingford forecast](spaghetti_rain_wallingford.png)

### How do I understand the plots?
These are 10-day forecasts of the river height differential (related to the flow speed) for the Isis, Godstow, and Wallingford stretches. The river is most likely to follow the purple (mean) trajectory, but we could end up on any of the blue ones depending on how much rain we actually get. Plots update automatically every hour, with new rainfall forecasts incorporated as soon as they are released.

### Where is the data from?
The Environment Agency's API provides the historical rainfall and the historical differential levels on which the model has been trained. This is also where the data from the last 10 days comes from, as shown on the plots. The rainfall forecasts come from the European Center for Medium Range Weather Forecasts' AI Forecasting System (ECWMF-AIFS). If you wish to clone the GitHub repo, you can very easily change the forecast data source.

### Are there any known issues?
The forecast is definitely not perfect! Occasionally I write about issues with the model and improvements I am hoping to make here: https://robertdoanesolomon.substack.com/. If there are any other problems, feel free to open an issue on GitHub.
