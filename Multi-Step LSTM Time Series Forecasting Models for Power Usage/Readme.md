# Multi-Step LSTM Time Series Forecasting Models for Power Usage

### Overview
This tutorial is divided into nine parts; they are:

- Problem Description
- Load and Prepare Dataset
- Model Evaluation
- LSTMs for Multi-Step Forecasting
- LSTM Model With Univariate Input and Vector Output
- Encoder-Decoder LSTM Model With Univariate Input
- Encoder-Decoder LSTM Model With Multivariate Input
- CNN-LSTM Encoder-Decoder Model With Univariate Input
- ConvLSTM Encoder-Decoder Model With Univariate Input

# Problem Description
The [Household Power Consumption](https://github.com/fahimalshihab/DATA-SCIENCE/blob/main/Multi-Step%20LSTM%20Time%20Series%20Forecasting%20Models%20for%20Power%20Usage/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip) dataset is a multivariate time series dataset that describes the electricity consumption for a single household over four years.
The data was collected between December 2006 and November 2010 and observations of power consumption within the household were collected every minute.

It is a multivariate series comprised of seven variables (besides the date and time); they are:

- global_active_power: The total active power consumed by the household (kilowatts).
- global_reactive_power: The total reactive power consumed by the household (kilowatts).
- voltage: Average voltage (volts).
- global_intensity: Average current intensity (amps).
- sub_metering_1: Active energy for kitchen (watt-hours of active energy).
- sub_metering_2: Active energy for laundry (watt-hours of active energy).
- sub_metering_3: Active energy for climate control systems (watt-hours of active energy).

# Load and Prepare Dataset

- “household_power_consumption.txt” that is about 127 megabytes in size and contains all of the observations.

  We can use the read_csv() function to load the data and combine the first two columns into a single date-time column that we can use as an index.
<code>
# load all data
dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
</code>

- Next, we can mark all missing values indicated with a ‘?‘ character with a NaN value, which is a float.

  This will allow us to work with the data as one array of floating point values rather than mixed types (less efficient.)
<code>
# mark all missing values
dataset.replace('?', nan, inplace=True)
# make dataset numeric
dataset = dataset.astype('float32')
</code>

- We also need to fill in the missing values now that they have been marked.

 A very simple approach would be to copy the observation from the same time the day before. We can implement this in a function named fill_missing() that will take the NumPy array of the data and copy values from exactly 24 
 hours ago.
<code>
# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]
</code>

- We can apply this function directly to the data within the DataFrame.

<code>
  # fill missing
fill_missing(dataset.values)
</code>

- Now we can create a new column that contains the remainder of the sub-metering, using the calculation from the previous section.

<code>
  # add a column for for the remainder of sub metering
values = dataset.values
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
</code>

- We can now save the cleaned-up version of the dataset to a new file; in this case we will just change the file extension to .csv and save the dataset as ‘household_power_consumption.csv‘.

  <code>
    # save updated dataset
dataset.to_csv('household_power_consumption.csv')
  </code>

#### Tying all of this together, the complete example of loading, cleaning-up, and saving the dataset is listed below.
```
# load and clean-up data
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric

# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

# load all data
dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# mark all missing values
dataset.replace('?', nan, inplace=True)
# make dataset numeric
dataset = dataset.astype('float32')
# fill missing
fill_missing(dataset.values)
# add a column for for the remainder of sub metering
values = dataset.values
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
# save updated dataset
dataset.to_csv('household_power_consumption.csv')
```

# Model Evaluation
In this section, we will consider how we can develop and evaluate predictive models for the household power dataset.

This section is divided into four parts; they are:

- Problem Framing
- Evaluation Metric
- Train and Test Sets
- Walk-Forward Validation

  ## Problem Framing
  we will use the data to explore a very specific question; that is:

<h2>" Given recent power consumption, what is the expected power consumption for the week ahead ? "</h2>

This requires that a predictive model forecast the total active power for each day over the next seven days.

Technically, this framing of the problem is referred to as a multi-step time series forecasting problem, given the multiple forecast steps. A model that makes use of multiple input variables may be referred to as a multivariate multi-step time series forecasting model.

A model of this type could be helpful within the household in planning expenditures. It could also be helpful on the supply side for planning electricity demand for a specific household.

This framing of the dataset also suggests that it would be useful to downsample the per-minute observations of power consumption to daily totals. This is not required, but makes sense, given that we are interested in total power per day.


-  We can achieve this easily using the resample() function on the pandas DataFrame. Calling this function with the argument ‘D‘ allows the loaded data indexed by date-time to be grouped by day 
   (see all offset aliases). We can then calculate the sum of all observations for each day and create a new dataset of daily power consumption data for each of the eight variables.

    The complete example is listed below.

 <code>
# resample minute data to total for each day
from pandas import read_csv
# load the new file
dataset = read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# resample data to daily
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()
# summarize
print(daily_data.shape)
print(daily_data.head())
# save
daily_data.to_csv('household_power_consumption_days.csv')
 </code>

- Running the example creates a new daily total power consumption dataset and saves the result into a separate file named ‘household_power_consumption_days.csv‘.

  We can use this as the dataset for fitting and evaluating predictive models for the chosen framing of the problem.

 ## Evaluation Metric

<pre>
A forecast will be comprised of seven values, one for each day of the week ahead.

It is common with multi-step forecasting problems to evaluate each forecasted time step separately. This is helpful for a few reasons:

To comment on the skill at a specific lead time (e.g. +1 day vs +3 days).
To contrast models based on their skills at different lead times (e.g. models good at +1 day vs models good at days +5).
The units of the total power are kilowatts and it would be useful to have an error metric that was also in the same units. Both Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) fit this bill, although RMSE is more commonly used and will be adopted in this tutorial. Unlike MAE, RMSE is more punishing of forecast errors.

The performance metric for this problem will be the RMSE for each lead time from day 1 to day 7.

As a short-cut, it may be useful to summarize the performance of a model using a single score in order to aide in model selection.

One possible score that could be used would be the RMSE across all forecast days.

The function evaluate_forecasts() below will implement this behavior and return the performance of a model based on multiple seven-day forecasts.
</pre>
 

<code>
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
</code>


Running the function will first return the overall RMSE regardless of day, then an array of RMSE scores for each day.

## Train and Test Sets

<pre>
We will use the first three years of data for training predictive models and the final year for evaluating models.

The data in a given dataset will be divided into standard weeks. These are weeks that begin on a Sunday and end on a Saturday.

This is a realistic and useful way for using the chosen framing of the model, where the power consumption for the week ahead can be predicted. It is also helpful with modeling, where models can be used to predict a specific day (e.g. Wednesday) or the entire sequence.

We will split the data into standard weeks, working backwards from the test dataset.

The final year of the data is in 2010 and the first Sunday for 2010 was January 3rd. The data ends in mid November 2010 and the closest final Saturday in the data is November 20th. This gives 46 weeks of test data.

The first and last rows of daily data for the test dataset are provided below for confirmation.

The daily data starts in late 2006.

The first Sunday in the dataset is December 17th, which is the second row of data.

Organizing the data into standard weeks gives 159 full standard weeks for training a predictive model.
The function split_dataset() below splits the daily data into train and test sets and organizes each into standard weeks.

Specific row offsets are used to split the data using knowledge of the dataset. The split datasets are then organized into weekly data using the NumPy split() function.

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

We can test this function out by loading the daily dataset and printing the first and last rows of data from both 
the train and test sets to confirm they match the expectations above.

	
</pre>

#### The complete code example is listed below.



```
# split into standard weeks
from numpy import split
from numpy import array
from pandas import read_csv

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
train, test = split_dataset(dataset.values)
# validate train data
print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])
# validate test
print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])
```

Running the example shows that indeed the train dataset has 159 weeks of data, whereas the test dataset has 46 weeks.

We can see that the total active power for the train and test dataset for the first and last rows match the data for the specific dates that we defined as the bounds on the standard weeks for each set.

## Walk-Forward Validation

Models will be evaluated using a scheme called walk-forward validation.

This is where a model is required to make a one week prediction, then the actual data for that week is made available to the model so that it can be used as the basis for making a prediction on the subsequent week. This is both realistic for how the model may be used in practice and beneficial to the models allowing them to make use of the best available data.

We can demonstrate this below with separation of input data and output/predicted data.

The walk-forward validation approach to evaluating predictive models on this dataset is provided below named evaluate_model().

The train and test datasets in standard-week format are provided to the function as arguments. An additional argument n_input is provided that is used to define the number of prior observations that the model will use as input in order to make a prediction.

Two new functions are called: one to build a model from the training data called build_model() and another that uses the model to make forecasts for each new standard week called forecast(). These will be covered in subsequent sections.

We are working with neural networks, and as such, they are generally slow to train but fast to evaluate. This means that the preferred usage of the models is to build them once on historical data and to use them to forecast each step of the walk-forward validation. The models are static (i.e. not updated) during their evaluation.

This is different to other models that are faster to train where a model may be re-fit or updated each step of the walk-forward validation as new data is made available. With sufficient resources, it is possible to use neural networks this way, but we will not in this tutorial.

The complete evaluate_model() function is listed below.
	
```
# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
```

Once we have the evaluation for a model, we can summarize the performance.

The function below named summarize_scores() will display the performance of a model as a single line for easy comparison with other models

```
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
```
 We now have all of the elements to begin evaluating predictive models on the dataset.

# LSTMs for Multi-Step Forecasting
Recurrent neural networks, or RNNs, are specifically designed to work, learn, and predict sequence data.

A recurrent neural network is a neural network where the output of the network from one time step is provided as an input in the subsequent time step. This allows the model to make a decision as to what to predict based on both the input for the current time step and direct knowledge of what was output in the prior time step.

Perhaps the most successful and widely used RNN is the long short-term memory network, or LSTM for short. It is successful because it overcomes the challenges involved in training a recurrent neural network, resulting in stable models. In addition to harnessing the recurrent connection of the outputs from the prior time step, LSTMs also have an internal memory that operates like a local variable, allowing them to accumulate state over the input sequence.
LSTMs offer a number of benefits when it comes to multi-step time series forecasting; they are:

Native Support for Sequences. LSTMs are a type of recurrent network, and as such are designed to take sequence data as input, unlike other models where lag observations must be presented as input features.
Multivariate Inputs. LSTMs directly support multiple parallel input sequences for multivariate inputs, unlike other models where multivariate inputs are presented in a flat structure.
Vector Output. Like other neural networks, LSTMs are able to map input data directly to an output vector that may represent multiple output time steps.
Further, specialized architectures have been developed that are specifically designed to make multi-step sequence predictions, generally referred to as sequence-to-sequence prediction, or seq2seq for short. This is useful as multi-step time series forecasting is a type of seq2seq prediction.

An example of a recurrent neural network architecture designed for seq2seq problems is the encoder-decoder LSTM.

An encoder-decoder LSTM is a model comprised of two sub-models: one called the encoder that reads the input sequences and compresses it to a fixed-length internal representation, and an output model called the decoder that interprets the internal representation and uses it to predict the output sequence.

The encoder-decoder approach to sequence prediction has proven much more effective than outputting a vector directly and is the preferred approach.

Generally, LSTMs have been found to not be very effective at auto-regression type problems. These are problems where forecasting the next time step is a function of recent time steps.

One-dimensional convolutional neural networks, or CNNs, have proven effective at automatically learning features from input sequences.

A popular approach has been to combine CNNs with LSTMs, where the CNN is as an encoder to learn features from sub-sequences of input data which are provided as time steps to an LSTM. This architecture is called a CNN-LSTM.
A power variation on the CNN LSTM architecture is the ConvLSTM that uses the convolutional reading of input subsequences directly within an LSTM’s units. This approach has proven very effective for time series classification and can be adapted for use in multi-step time series forecasting.

In this tutorial, we will explore a suite of LSTM architectures for multi-step time series forecasting. Specifically, we will look at how to develop the following models:

LSTM model with vector output for multi-step forecasting with univariate input data.
Encoder-Decoder LSTM model for multi-step forecasting with univariate input data.
Encoder-Decoder LSTM model for multi-step forecasting with multivariate input data.
CNN-LSTM Encoder-Decoder model for multi-step forecasting with univariate input data.
ConvLSTM Encoder-Decoder model for multi-step forecasting with univariate input data.

The models will be developed and demonstrated on the household power prediction problem. A model is considered skillful if it achieves performance better than a naive model, which is an overall RMSE of about 465 kilowatts across a seven day forecast.

We will not focus on the tuning of these models to achieve optimal performance; instead, we will stop short at skillful models as compared to a naive forecast. The chosen structures and hyperparameters are chosen with a little trial and error. The scores should be taken as just an example rather than a study of the optimal model or configuration for the problem.

Given the stochastic nature of the models, it is good practice to evaluate a given model multiple times and report the mean performance on a test dataset. In the interest of brevity and keeping the code simple, we will instead present single-runs of models in this tutorial.

We cannot know which approach will be the most effective for a given multi-step forecasting problem. It is a good idea to explore a suite of methods in order to discover what works best on your specific dataset.



