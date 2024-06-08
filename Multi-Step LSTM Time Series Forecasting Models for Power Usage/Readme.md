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




















