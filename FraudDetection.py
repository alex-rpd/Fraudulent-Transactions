import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


class FraudulentData:
    """
    Main class that contains the code for the Fraudulent transactions model
    """
    def __init__(self):
        """
        Function that loads the transactions and label files and merges them
        on the eventId column
        :return: pandas DataFrame self.df
        """
        self.df = pd.read_csv('./data/data_anonymised.csv')
        self.postcodes = pd.read_excel('./data/postcodes_regions.xlsx')

    def create_nn_model(self, inputs_number):
        """
        A function that will create a NN Sequential model.

        The model is comprised of an input Dense layer of input_number units
        and an output Dense layer with a single unit.

        There are 4 hidden layers, each having a 40% Dropout layer.

        Each layer is using a tanh activation function.

        The output layer has a sigmoid activation function.

        :param inputs_number: the number of the initial units for
                              the input layer, corresponds to the number of
                              features in the dataset
        :return: the tf.keras model
        """
        self.oversample_model = Sequential()

        self.oversample_model.add(
            Dense(inputs_number, input_shape=(inputs_number,),
                  activation='tanh'))
        self.oversample_model.add(Dropout(0.4))

        self.oversample_model.add(Dense(inputs_number, activation='tanh'))
        self.oversample_model.add(Dropout(0.4))

        self.oversample_model.add(Dense(inputs_number, activation='tanh'))
        self.oversample_model.add(Dropout(0.4))

        self.oversample_model.add(Dense(inputs_number, activation='tanh'))
        self.oversample_model.add(Dropout(0.4))

        self.oversample_model.add(Dense(64, activation='tanh'))
        self.oversample_model.add(Dropout(0.4))

        self.oversample_model.add(Dense(1, activation='sigmoid'))
        print(self.oversample_model.summary())


    def process_data(self):
        """
        A function that will process the df pandas DataFrame:
            - Will convert the datetime columns and create the part of day bins
            - Will processes the merchant zip codes and group the countries
            - Will create the dummy columns
            - Will drop columns that serve to purpose to the model
            - Will create the self.dum_data DataFrame that is the basis for the
              X - y split
        :return:
        """
        print('Process started...')

        # create a Fraud flag
        self.df['fraud'] = 0
        self.df.loc[self.df['reported_time'].notna(), 'fraud'] = 1

        # convert to datetime and create the time bins
        self.df['transaction_time'] = pd.to_datetime(self.df['transaction_time'])
        self.df['transaction_time_time'] = self.df['transaction_time'].dt.time
        self.df['transaction_time_month'] = self.df['transaction_time'].dt.month
        self.df['transaction_time_year'] = self.df['transaction_time'].dt.year
        self.df['bins'] = pd.cut(self.df['transaction_time'].dt.hour,
                                 bins=[2,
                                       8,
                                       14,
                                       20, ],
                                 labels=['02-08',
                                         '08-14',
                                         '14-20'],
                                 include_lowest=True).values.add_categories(
            '20-02').fillna('20-02')

        # format the merchant zip codes based on the country
        # create the UK region specifics
        self.df['match'] = self.df[self.df['merchant_zip'].notna()][
            'merchant_zip'].apply(
            lambda x: re.match(pattern=r'[a-zA-Z]{1,2}', string=x))
        self.df['post_code_start'] = self.df[self.df['match'].notna()][
            'match'].apply(
            lambda x: x[0])
        self.df['uk_region'] = self.df[self.df['post_code_start'].notna()][
            'post_code_start'].str.upper().map(
            self.postcodes.set_index('Prefix')['Region'].to_dict())
        self.df.loc[(self.df['match'].isna()) &
                    (self.df['merchant_country'] == 826),
                    'uk_region'] = 'other'
        self.df.loc[(self.df['uk_region'].isna()) &
                    (self.df['merchant_country'] == 826),
                    'uk_region'] = 'other'
        self.df['country_for_dummies'] = self.df[
            self.df['uk_region'].notna()].apply(
            lambda x: str(x['merchant_country']) + '_' + x['uk_region'].replace(
                ' ',
                ''),
            axis=1)
        self.df.loc[self.df['merchant_country'] == 442,
                    'country_for_dummies'] = 'country_442'
        self.df.loc[self.df['merchant_country'] == 840,
                    'country_for_dummies'] = 'country_840'
        self.df.loc[self.df['merchant_country'] == 372,
                    'country_for_dummies'] = 'country_372'
        self.df.loc[self.df['country_for_dummies'].isna(),
                    'country_for_dummies'] = 'country_other'

        # columns to drop - no added value to the model
        drop_cols = ['post_code_start', 'merchant_zip',
                     'match', 'uk_region', 'transaction_time_year',
                     'transaction_time_month', 'transaction_time', 'event_id',
                     'account_number', 'merchant_id', 'merchant_country',
                     'reported_time', 'transaction_time_time']
        self.data = self.df.drop(columns=drop_cols)

        # columns to create the dummy variables
        dum_cols = ['pos_entry_mode', 'merchant_category',
                    'bins', 'country_for_dummies']
        self.data_dum = pd.get_dummies(self.data, columns=dum_cols,
                                       drop_first=True)

    def define_X_y(self):
        """
        Function that defines the X and y datasets, as well as the
        oversampled SMOTE training data
        :return:
        """
        # define the training X and testing y data splits
        X = self.data_dum.drop('fraud', axis=1)
        y = self.data_dum['fraud']
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y,
                             test_size=0.3,
                             random_state=42)

        # Use a min-max scaling for the data
        # Apply the fit and transform on the training but only transform the
        # testing to avoid data leaking issues
        min_max_scaler = MinMaxScaler()
        self.X_train = min_max_scaler.fit_transform(self.X_train)
        self.X_test = min_max_scaler.transform(self.X_test)

        # define a SMOTE object - a minority class oversampling approach
        # research on simple 50-50 split based on the minority class (fraudulent
        # transactions, i.e. the 875 Fraudulent and 875 random non Fraudulent
        # didn't prove to be as good as a SMOTE & NN approach
        sm = SMOTE(random_state=42)
        self.Xsm_train, self.ysm_train = sm.fit_sample(self.X_train,
                                                       self.y_train)

    def train_model(self):
        """
        This function will create, compile and train a fully connected
        Deep Neural Network using a Stochastic Gradient Descent optimiser. The
        network is trained for 1200 epochs that are sufficient to prevent
        overfitting issues.
        :return:
        """
        n_inputs = self.Xsm_train.shape[1]

        # create the neural network model
        self.create_nn_model(inputs_number=n_inputs)

        # complite the model
        self.oversample_model.compile(optimizer=SGD(learning_rate=0.001),
                                      loss='binary_crossentropy',
                                      metrics=['accuracy'])

        # train the network on the oversampled data
        # monitor the validation performance (loss and accuracy) on the original
        # test datasets.
        # Training set at 1200 epochs, enough to lower the validation losses to
        # ~20% with validation accuracy > 90%. No overfitting phenomena observed
        self.oversample_model.fit(self.Xsm_train, self.ysm_train,
                                  validation_data=(self.X_test, self.y_test),
                                  batch_size=128, epochs=1200, verbose=2)

    def predict_and_export(self):
        """
        This function will create a losses dataframe based on the previously
        trained model and export it in a pickle for future analysis and
        plotting of the training performance.
        The main scores of the model will be printed based on the predictions
        and the model will be saved in the current folder, so it can be used
        for future predictions directly, without the need to retrain.
        :return:
        """
        # create a losses dataframe and export to pickle
        losses = pd.DataFrame(self.oversample_model.history.history)
        losses.to_pickle('model_losses_dataframe.p')

        # predict classes based on a 70% confidence
        # this value with adjust the actual scores below
        self.oversample_predictions = self.oversample_model.predict(self.X_test)
        self.oversample_fraud_predictions = (
                self.oversample_model.predict(self.X_test) > 0.7).astype("int32")

        print('Precision score: ', metrics.precision_score(y_true=self.y_test,
                                                           y_pred=self.oversample_fraud_predictions))
        print('Accuracy score: ', metrics.accuracy_score(y_true=self.y_test,
                                                         y_pred=self.oversample_fraud_predictions))
        print('Recall score: ', metrics.recall_score(y_true=self.y_test,
                                                     y_pred=self.oversample_fraud_predictions))
        print('F1 score: ', metrics.f1_score(y_true=self.y_test,
                                             y_pred=self.oversample_fraud_predictions))

        # save the trained model so there is no need to re-run
        model_name = 'nn_model_seq_5l'
        self.oversample_model.save(model_name)
        print(f'Model exported as: {model_name}')

    def run_all(self):
        """
        Aggregating function that will run the model end-to-end in a single
        step.
        All the data, the model and model outcomes are saved as attributes of
        the created object when the class is called:
            e.g. using fraud = FraudulentData()
                       fraud.run_all() # will run the model end to end
                       df = fraud.df # will return the formatted dataframe
                       model = fraud.oversample_model # will return the model
        :return:
        """
        self.process_data()
        self.define_X_y()
        self.train_model()
        self.predict_and_export()


if __name__ == '__main__':
    # call example for the class and running end-to-end
    fraud = FraudulentData()
    fraud.run_all()
