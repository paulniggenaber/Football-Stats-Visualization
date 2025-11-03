# Predicting football transfer fees based on individual player data

By combining indidvidual player data from fbref.com scraped in the context of the project Football-Player-Performance with infos on transfer fees a database of 6000 - 7700 rows is created.
Transfers from the top 5 european leagues as well as the Chmapionship, Eredivise and Ligue NOS in the timespan of 2017 - 2022 are included. 
This time restriction's lower bound comes from the introduction of extensive data collection in football, including andvances metrics like expected goals.
The upper bound is set by available transfer data on the internet.
The final dataframe contains an aggregation of basic transfer infos like fee and age at transfer with aggregated performance data looking back at the time before the transfer happended.
I am looking back on 1 - 4 season halves of individual player performance, respectively. Furthermore subsets depending on a players position are created containing only the more relevant stats for the given position.

Using this data various ML models are trained and their predcitive power is compared.

Furthermore the limitations of the data and problems when it comes to predicting transfer fees in football are discussed.

- data_preprocessing.ipynb contains the process of preparing the data for ML
- model_training.ipynb examines the predictive power and compares models giving a reasoned choice of models and discusses possible limitations preventing better predictive power
- BDassignment2.pdf contains the assignment for this project providing context
