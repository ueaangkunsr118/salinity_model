import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

def main():
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False
    csv_file1 = 'cleaned_ladpo_v2.csv'
    df1 = pd.read_csv(csv_file1)
    df1['date_time'] = pd.to_datetime(df1['date_time'], format='%Y-%m-%d %H:%M:%S')
    df1.set_index('date_time', inplace=True)
    df1 = df1.loc[:, ['Salinity', 'Sensor_Depth', 'Temperature']].dropna()
    df1 = df1[df1['Sensor_Depth'] > -10]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df1.index, df1['Salinity'], '.', markersize=0.5, label='Salinity', color='b')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Salinity', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(df1.index, df1['Sensor_Depth'], '.', markersize=0.3, label='Sensor Depth', color='r')
    ax2.set_ylabel('Sensor Depth', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax3 = ax1.twinx()
    ax3.plot(df1.index, df1['Temperature'], '.', markersize=0.3, label='Temperature', color='g')
    ax3.set_ylabel('Temperature', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    plt.title('Salinity and Scaled Sensor Depth Over Time (Outliers Removed)')
    plt.show()
    csv_file2 = 'cleaned_faifa_v2.csv'
    df2 = pd.read_csv(csv_file2)
    df2['date_time'] = pd.to_datetime(df2['date_time'], format='%Y-%m-%d %H:%M:%S')
    df2.set_index('date_time', inplace=True)
    df2 = df2.loc[: , ['Salinity', 'Sensor_Depth', 'Temperature']].dropna()
    df2 = df2[(df2['Salinity'] > 0) & (df2['Salinity'] < 50)]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df2.index, df2['Salinity'], '.', markersize=0.5, label='Salinity', color='b')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Salinity', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(df2.index, df2['Sensor_Depth'], '.', markersize=0.3, label='Sensor Depth', color='r')
    ax2.set_ylabel('Sensor Depth', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax3 = ax1.twinx()
    ax3.plot(df2.index, df2['Temperature'], '.', markersize=0.3, label='Temperature', color='g')
    ax3.set_ylabel('Temperature', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    df = df2.loc['2019-01-01':, ['Salinity']].dropna()
    plt.title('Salinity, Sensor Depth, and Temp Over Time (Outliers Removed)')
    plt.show()
    csv_file3 = 'cleaned_port_v2.csv'
    df3 = pd.read_csv(csv_file3)
    df3['date_time'] = pd.to_datetime(df3['date_time'], format='%Y-%m-%d %H:%M:%S')
    df3.set_index('date_time', inplace=True)
    df3 = df3.loc[:, ['Salinity', 'Sensor_Depth', 'Temperature']].dropna()
    df3 = df3[df3['Sensor_Depth'] > -10]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df3.index, df3['Salinity'], '.', markersize=0.5, label='Salinity', color='b')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Salinity', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(df3.index, df3['Sensor_Depth'], '.', markersize=0.3, label='Sensor Depth', color='r')
    ax2.set_ylabel('Sensor Depth', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax3 = ax1.twinx()
    ax3.plot(df3.index, df3['Temperature'], '.', markersize=0.3, label='Temperature', color='g')
    ax3.set_ylabel('Temperature', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    plt.title('Salinity and Scaled Sensor Depth Over Time (Outliers Removed)')
    plt.show()
    column_indices = {name: i for i, name in enumerate(df1.columns)}
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    num_features = df.shape[1]
    val_df = (val_df - train_mean) / train_std  
    test_df = (test_df - train_mean) / train_std 
    df_plot = (train_df)
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df_plot)
    plt.xlabel('Column')
    plt.ylabel('Normalized')
    plt.title('Violin Plot of Normalized Salinity and Sensor Depth')
    plt.show()
    class WindowGenerator():
      def __init__(self, input_width, label_width, shift,
                   train_df=train_df, val_df=val_df, test_df=test_df,
                   label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
      def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                         label_columns=['Salinity'])
    w1
    def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])
      return inputs, labels
    WindowGenerator.split_window = split_window
    example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                               np.array(train_df[100:100+w1.total_window_size]),
                               np.array(train_df[200:200+w1.total_window_size])])
    example_inputs, example_labels = w1.split_window(example_window)
    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'Labels shape: {example_labels.shape}')
    w1.example = example_inputs, example_labels
    def plot(self, model=None, plot_col='Salinity', max_subplots=3):
      inputs, labels = self.example
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)
        if self.label_columns:
          label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
          label_col_index = plot_col_index
        if label_col_index is None:
          continue
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
        if model is not None:
          predictions = model(inputs)
          plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
        if n == 0:
          plt.legend()
      plt.xlabel('Date')
    WindowGenerator.plot = plot
    w1.plot()
    def make_dataset(self, data):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.utils.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=False,
          batch_size=32,)
      ds = ds.map(self.split_window)
      return ds
    WindowGenerator.make_dataset = make_dataset
    @property
    def train(self):
      return self.make_dataset(self.train_df)
    @property
    def val(self):
      return self.make_dataset(self.val_df)
    @property
    def test(self):
      return self.make_dataset(self.test_df)
    @property
    def example(self):
      """Get and cache an example batch of `inputs, labels` for plotting."""
      result = getattr(self, '_example', None)
      if result is None:
        result = next(iter(self.train))
        self._example = result
      return result
    WindowGenerator.train = train
    WindowGenerator.val = val
    WindowGenerator.test = test
    WindowGenerator.example = example
    w1.train.element_spec
    for example_inputs, example_labels in w1.train.take(1):
      print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
      print(f'Labels shape (batch, time, features): {example_labels.shape}')
    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1,
        label_columns=['Salinity'])
    single_step_window
    for example_inputs, example_labels in single_step_window.train.take(1):
      print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
      print(f'Labels shape (batch, time, features): {example_labels.shape}')
    class Baseline(tf.keras.Model):
      def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
      def call(self, inputs):
        if self.label_index is None:
          return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    baseline = Baseline(label_index=column_indices['Salinity'])
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])
    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)
    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        label_columns=['Salinity'])
    wide_window
    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', baseline(wide_window.example[0]).shape)
    wide_window.plot(baseline)
    MAX_EPOCHS = 20
    def compile_and_fit(model, window, patience=2):
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')
      model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])
      history = model.fit(window.train, epochs=MAX_EPOCHS,
                          validation_data=window.val,
                          callbacks=[early_stopping])
      return history
    lstm_model16 = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model16, wide_window)
    IPython.display.clear_output()
    val_performance['LSTM16'] = lstm_model16.evaluate(wide_window.val, return_dict=True)
    performance['LSTM16'] = lstm_model16.evaluate(wide_window.test, verbose=0, return_dict=True)
    wide_window.plot(lstm_model16)
    lstm_model32 = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model32, wide_window)
    IPython.display.clear_output()
    val_performance['LSTM32'] = lstm_model32.evaluate(wide_window.val, return_dict=True)
    performance['LSTM32'] = lstm_model32.evaluate(wide_window.test, verbose=0, return_dict=True)
    wide_window.plot(lstm_model32)
    lstm_model64 = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model64, wide_window)
    IPython.display.clear_output()
    val_performance['LSTM64'] = lstm_model64.evaluate(wide_window.val, return_dict=True)
    performance['LSTM64'] = lstm_model64.evaluate(wide_window.test, verbose=0, return_dict=True)
    wide_window.plot(lstm_model64)
    lstm_model128 = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model128, wide_window)
    IPython.display.clear_output()
    val_performance['LSTM128'] = lstm_model128.evaluate(wide_window.val, return_dict=True)
    performance['LSTM128'] = lstm_model128.evaluate(wide_window.test, verbose=0, return_dict=True)
    wide_window.plot(lstm_model128)
    lstm_model256 = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model256, wide_window)
    IPython.display.clear_output()
    val_performance['LSTM128'] = lstm_model256.evaluate(wide_window.val, return_dict=True)
    performance['LSTM128'] = lstm_model256.evaluate(wide_window.test, verbose=0, return_dict=True)
    wide_window.plot(lstm_model256)
    lstm_model512 = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(521, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model512, wide_window)
    IPython.display.clear_output()
    val_performance['LSTM521'] = lstm_model512.evaluate(wide_window.val, return_dict=True)
    performance['LSTM521'] = lstm_model512.evaluate(wide_window.test, verbose=0, return_dict=True)
    wide_window.plot(lstm_model512)
    lstm_model1024 = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(1024, return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model1024, wide_window)
    IPython.display.clear_output()
    val_performance['LSTM1024'] = lstm_model1024.evaluate(wide_window.val, return_dict=True)
    performance['LSTM1024'] = lstm_model1024.evaluate(wide_window.test, verbose=0, return_dict=True)
    wide_window.plot(lstm_model1024)
    cm = lstm_model.metrics[1]
    cm.metrics
    val_performance
    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in val_performance.values()]
    test_mae = [v[metric_name] for v in performance.values()]
    plt.ylabel('mean_absolute_error [Salinity, normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
               rotation=45)
    _ = plt.legend()
    for name, value in performance.items():
      print(f'{name:12s}: {value[metric_name]:0.4f}')
    with open('performance_results.txt', 'w') as file:
        for name, value in performance.items():
            file.write(f'{name:12s}: {value[metric_name]:0.4f}\n')

if __name__ == "__main__":
    main()
