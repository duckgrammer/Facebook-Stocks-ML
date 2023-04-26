KEY = 'DFN1YT3C2J6YBUXI'
from alpha_vantage.timeseries import TimeSeries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
#from tensorflow.python.keras.optimizers import Adam
#import tensorflow.python.keras.optimizers as opts
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from keras.callbacks import History
import time


def download_data():
    ts = TimeSeries(key=KEY)
    data, meta_data = ts.get_daily_adjusted('META', outputsize='full')

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date]["5. adjusted close"]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output

data_date, data_close_price, num_data_points, display_date_range = download_data()

#graph the stock price
# fig = figure(figsize=(25,10),dpi=80)
# plt.plot(data_date,data_close_price,label="META Close Price")
xticks = [data_date[i] if ((i%90==0 and (num_data_points-i) > 90) or i==num_data_points-1) else None for i in range(num_data_points)]
x = np.arange(0,len(xticks))
# plt.xticks(x, xticks, rotation='vertical')
# plt.title("Daily close")
# plt.grid(b=None, which='major', axis='y', linestyle='--')
# plt.legend()
# plt.show()

#plot the train vs validation stuff
# split = round(num_data_points * .8)
# train = data_close_price[:split]
# val = data_close_price[split:]
# time1 = data_date[:split]
# time2 = data_date[split:]
# fig = figure(figsize=(25,10),dpi=80)
# plt.plot(time1,train, label="Prices (train)", color='#3D9970')
# plt.plot(time2,val, label="Prices (validation)", color='#0074D9')
# plt.xticks(x, xticks, rotation='vertical')
# plt.title("Splitting Training and Validation")
# plt.grid(b=None, which='major', axis='y', linestyle='--')
# plt.legend()
# plt.show()

#normalize data
mean = np.mean(data_close_price)
std = np.std(data_close_price)
normalized_data_close_price = (data_close_price - mean) / std

#format
data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=20)
data_y = prepare_data_y(normalized_data_close_price, window_size=20)

data_x = data_x.astype(np.float32)
data_y = data_y.astype(np.float32)
data_x = np.expand_dims(data_x,2)
data_x_unseen = data_x_unseen.astype(np.float32)
data_x_unseen = np.expand_dims(data_x_unseen,1)
data_x_unseen = data_x_unseen.T
data_x_unseen = np.expand_dims(data_x_unseen,-1)

#data x now 3 dims, y is 2 dims for both train and test
#could add validation area
split_index = int(data_y.shape[0]*.8)
x_train = data_x[:split_index]
x_val = data_x[split_index:]
y_train = data_y[:split_index]
y_val = data_y[split_index:]

# model = Sequential([layers.Input((20,1)),
#                      layers.LSTM(64),
#                      layers.Dense(32,activation='relu'),
#                      layers.Dropout(.2),
#                      layers.Dense(32, activation='relu'),
#                      layers.Dense(1)])
#
#
# model.compile(loss='mse',
#                        optimizer=Adam(learning_rate=.001),
#                        metrics=['MeanAbsolutePercentageError'])

# history = History()
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val),
#                    callbacks=[history], verbose=1)
#
#
#
# tomorrow_prediction = model.predict(data_x_unseen, batch_size=64, verbose=1)
# tomorrow_prediction = (tomorrow_prediction * std) + mean
# print('done')



#config model
dropout_rate_search = [.1,.2,.3]
learning_rate_search = [.0001,.001,.01]
num_epochs = 25

for i in range(0, 3):
    for j in range(0,3):
        dr = dropout_rate_search[i]
        lr = learning_rate_search[j]

        model = Sequential([layers.Input((20, 1)),
                            layers.LSTM(64),
                            layers.Dense(32, activation='relu'),
                            layers.Dropout(dr),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(1)])

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=lr),
                      metrics=['MeanAbsolutePercentageError'])

        start = time.time()
        history = History()
        model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_val, y_val),
                  callbacks=[history], verbose=1)
        end = time.time()
        duration = str(round(end - start)) + ' seconds'
        predictions = model.predict(x_val, batch_size=64, verbose=1)
        predictions = (predictions * std) + mean
        val_times = data_date[split_index + 20:]

        #get the results
        results = model.evaluate(x_val, y_val)
        print('results (loss/accuracy)' + str(results) + 'iteration: ' + str((j+1) * (i+1)))

        #get tomorrows price
        tomorrow_prediction = model.predict(data_x_unseen, batch_size=64, verbose=1)
        tomorrow_prediction = (tomorrow_prediction * std) + mean
        print('tomorrow price: ' + str(tomorrow_prediction) + 'iteration: ' + str((j+1) * (i+1)))

        # graph the predictions
        fig = figure(figsize=(25, 10), dpi=80)
        plt.plot(data_date, data_close_price, label="Dataset")
        plt.plot(val_times, predictions, label="Predicted prices", color='#3D9970')
        plt.xticks(x, xticks, rotation='vertical')
        plt.title("Predictions visualized" + duration + ',Dropout Rate: ' + str(dr) + ',Learning Rate: ' + str(lr))
        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.legend()
        #plt.savefig('predictions' + 'dr_' +str(dr) + 'lr_'+ str(lr)  + '.png')
        plt.clf()
        #plt.show()

        print("Dropout rate:" + str(dr))
        print("Learning rate: " + str(lr))
        print("Total time: " + duration)

        #plot the losses using mse
        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        epochs = range(1, num_epochs + 1)
        print(epochs)
        plt.plot(epochs, loss_train, 'g', label='Training loss (MSE)')
        plt.plot(epochs, loss_val, 'b', label='Validation loss (MSE)')
        plt.title("Config=" + duration + ',Dropout Rate: ' + str(dr) + ',Learning Rate: ' + str(lr))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.savefig('mse losses' + 'dr_' +str(dr) + 'lr_'+ str(lr) + '.png')
        plt.clf()
        #plt.show()

        #plot the mean absolute percentage errors
        mse_train = history.history['mean_absolute_percentage_error']
        mse_val = history.history['val_mean_absolute_percentage_error']
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, mse_train, 'g', label='Training MAPE')
        plt.plot(epochs, mse_val, 'b', label='Validation MAPE')
        plt.title("Config=" + duration + ',Dropout Rate: ' + str(dr) + ',Learning Rate: ' + str(lr))
        plt.xlabel('Epochs')
        plt.ylabel('Mean % error')
        plt.legend()
        #plt.savefig('mape' + 'dr_' +str(dr) + 'lr_'+ str(lr)  + '.png')
        plt.clf()
        #plt.show()

