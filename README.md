This is a Machine Learning practice exercise, applying [Catboost](https://catboost.ai/) to the Kaggle [Forest Fires Data Set Portugal dataset](https://www.kaggle.com/datasets/ishandutta/forest-fires-data-set-portugal) based on [this tutorial](https://towardsdatascience.com/an-end-to-end-machine-learning-project-heart-failure-prediction-part-1-ccad0b3b468a).

# The purpose

This is a small exercise is data science.


# The dataset

This data was used in P. Cortez and A. Morais, “A Data Mining Approach to Predict Forest Fires using Meteorological Data,” p. 12. of fires in Montesinho natural park.

![map of the park](img/img.png)

All points in the dataset refer to a fire within the park, those that have a area = 0 refer to events with burnt area <1 ha. We will consider these to be events where burnt = false and those with burnt area >=1 ha with burnt = true.

Each record refers to an x,y coordinate that designates a quadrant of the park, a month and a week day; it also incorporates metrics that are considered relevant to the likelihood of an event.

![monthly distribution](img/img.png)

![weekly distribution](img/img.png)

![geographical distribution](img/img.png)


# The model

We apply [Catboost](https://catboost.ai/) to understand the weights of the metrics.

We also attempt to figure out a "fire watch" vs "fire warning" ranges i.e. : what are the combined ranges at which fires are most likely and those where fires are all but certain.


# Suggestions of improvement

All these records are, in truth, positives which creates a survivor bias.
The original work refer to the expensiveness of active monitoring for fires and makes the case for passive monitoring of weather conditions.
Weather stations are becoming increasingly cheaper, which could be used for passive data collection and cross-referenced with after action fire reports
