This is a Machine Learning practice exercise, applying [Catboost](https://catboost.ai/) to the Kaggle [Forest Fires in Portugal dataset](https://www.kaggle.com/datasets/ishandutta/forest-fires-data-set-portugal) based on [this tutorial](https://towardsdatascience.com/an-end-to-end-machine-learning-project-heart-failure-prediction-part-1-ccad0b3b468a).


# The purpose

This is a small exercise is data science. Specifically, we will use the gradient boosting library [Catboost](https://catboost.ai/), which is especially interesting for heterogenous data. For more information see [Anna Veronika Dorogush' lecture](https://www.youtube.com/watch?v=usdEWSDisS0) at PyData London 2019 and the [notebook](https://github.com/catboost/tutorials/blob/master/events/2019_pydata_london/pydata_london_2019.ipynb).

By heterogenous data, we are refering to data points that have both numeric and categorical data (for instance, a person will have an age and an occupation).


# The dataset

This data was used in P. Cortez and A. Morais, “A Data Mining Approach to Predict Forest Fires using Meteorological Data,” p. 12. of fires in Montesinho natural park.

All points in the dataset refer to a fire within the park, those that have a area = 0 refer to events with burnt area <1 ha. We will consider these to be events where burnt = false and those with burnt area >=1 ha with burnt = true.

Each record refers to an x,y coordinate that designates a quadrant of the park, a month and a week day; it also incorporates metrics that are considered relevant to the likelihood of an event.


# The model

We apply Catboost to understand the weights of the metrics.


# The loss function
''Cross entropy'' is the recommended loss function for probabilistic data (i.e. multiple targets). Since this is a binary results (burnt / not burnt) we will use LogLoss.


# Some results

One feature of the Catboost package is feature importance, i.e. which feature is most relevant to the prediction.

![What features contribute most to an accurate prediction?](graph/importance_of_features.png?raw=true "What features contribute most to an accurate prediction?")

Month being second most important (more fires in summer months), the most important is day of the week. Looking as the distribution, we indeed see higher counts on Saturday and Sunday, more so if we include Friday and Monday. This might be related to human activity (more people visiting the park on weekends).

![Occurrences by day of the week](graph/occurrences_by_day.png?raw=true "Occurrences by day of the week")


# Future work

All these records are, in truth, positives (all had an active fire that results in burnt material) which creates a survivor bias. To improve this, we would need data points that are regularly recorded, not just when there are fires. As an example, "rain" is considered the least important predictor for the occurrence of a fire, but that is probably due to the fact that most events were recorded in August and September, when there is very little rain.

The original work mentions the cost of active monitoring of fires and makes the case for passive monitoring of weather conditions. Since weather stations (or some form of DIY using Arduino, for instance) are becoming increasingly cheaper, it would be possible to have passive data collection which could be cross-referenced with after action fire reports. With enough data, we could also attempt to figure out a "fire watch" vs "fire warning" ranges: what are the combined ranges at which fires are most likely and those where fires are all but certain.
