import json
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

"""
    This method loads the data for a specific station id, processes it and the train and evaluate
    a Random Forest Regressor algorithm to predict 3 different horizon. The function returns a  
"""
def evaluate(station_id, plot_gini_coef=False, use_time_features=False, show_plot=False):

    def index_to_hour(x):
        hour = x.index.strftime('%H')
        return hour
        pass

    def index_to_weekday(x):
        #Day_of_week = x.index.strftime('%a')
        Day_of_week = x.index.dayofweek
        return Day_of_week

        pass

    ## Load data
    df0 = load_data(station_id)

    ## Process data (Note: data could also be normalized and/or one-hot encoded for better results)
    if use_time_features:
        df_hours = df0.apply(index_to_hour)
        df_dayOfWeek = df0.apply(index_to_weekday)

        df0.insert(0, 'hour', df_hours.iloc[:, 0])
        df0.insert(0, 'Day_of_week', df_dayOfWeek.iloc[:, 0])
        pass

    ## Split the data into train and test sets (important keep the temporal order)
    #msk = np.random.rand(len(df0)) < 0.75
    #train_set_0 = df0[msk]
    #test_set_0 = df0[~msk]
    test_set_0, train_set_0 = train_test_split(df0, test_size=0.75, shuffle=False)

    ## Split the train_set into X_train (input data), y_train (output data)
    size_x = (df0.shape[1]-3)

    X_train_set = train_set_0.iloc[:, 0:size_x ]
    Y_train_set = train_set_0.iloc[:, size_x:df0.shape[1]]

    ## Split the test_set into X_test (input data), y_test (output data, aka ground truth)
    X_test_set = test_set_0.iloc[:, 0:size_x]
    Y_test_set = test_set_0.iloc[:, size_x:df0.shape[1]]

    ## Initialize the ML model
    RF = RandomForestRegressor()

    ## Train the model using the X_test and y_test data
    RF.fit(X_train_set, Y_train_set)


    ## Perform the predictions for the X_test set
    listOfList = RF.predict(X_test_set)

    ## Convert the received predictions (list of lists) to a dataframe (specify columns names)
    dfPredic = pandas.DataFrame(listOfList)
    dfPredic_int = dfPredic.round(0).astype(int)
    ## Compare the obtained predictions and the ground_truth(y_test_set)
    #mean_absolute_error = [0, 0, 0]
    from sklearn.metrics import mean_absolute_error
    mean_absolute_error = mean_absolute_error(Y_test_set, dfPredic_int, multioutput='raw_values')
    print("Mean absolute error: {}".format(mean_absolute_error))

    ## Plot Mean Absolute Error
    if show_plot:
        plt.bar(range(3), mean_absolute_error)

        plt.title('Mean absolute error (Station <{}>)'.format(station_id))
        plt.tight_layout()
        plt.show()

    ##Plot Gini coefficients and their importance
    if show_plot:
        importances = RF.feature_importances_

        plt.tight_layout()
        plt.title('Gini coefficients (Station <{}>)'.format(station_id))
        plt.bar(range(size_x), importances)
        plt.show()

    return mean_absolute_error


""" 
    Load the data from a file and return it as a panda dataframe 
    Columns:  'bikes(t-20)', 'bikes(t-15)', 'bikes(t-10)', 'bikes(t-5)',  ## (int) number of bikes present at the station t minutes ago
       'holiday_type(t)',                                                 ## (int) the type of holiday (none, school, public)  
       'client_arrivals(t)', 'client_departures(t)',                      ## (int) the number of departures/arrivals at this station during the last 5 minutes
       'cur_temperature(t)', 'cur_humidity(t)', 'cur_cloudiness(t)',      ## (float) the current weather information  
       'cur_wind(t)', 
       'f3h_temperature(t)', 'f3h_humidity(t)', 'f3h_cloudiness(t)',      ## (float) the 3h forecasted weather information
       'f3h_wind(t)', 
       'client_departures_s43(t-20)', 'client_departures_s43(t-15)',      ## (int) the number of departures at the neighbouring stations (s_##) t minutes ago
       'client_departures_s43(t-10)',
       'client_departures_s43(t-5)', 'client_departures_s43(t)',
       'client_departures_s12(t-20)', 'client_departures_s12(t-15)',
       'client_departures_s12(t-10)', 'client_departures_s12(t-5)',
       'client_departures_s12(t)', 'client_departures_s22(t-20)',
       'client_departures_s22(t-15)', 'client_departures_s22(t-10)',
       'client_departures_s22(t-5)', 'client_departures_s22(t)',
       'client_departures_s15(t-20)', 'client_departures_s15(t-15)',
       'client_departures_s15(t-10)', 'client_departures_s15(t-5)',
       'client_departures_s15(t)', 'client_departures_s36(t-20)',
       'client_departures_s36(t-15)', 'client_departures_s36(t-10)',
       'client_departures_s36(t-5)', 'client_departures_s36(t)', 
       'bikes(t)',                                                      ## The current number of bikes at the station
       'bikes(t+15)', 'bikes(t+30)', 'bikes(t+60)'                      ## The future number of bikes at the station (those are the values to predict)
    
"""
def load_data(station_id):
    with open("data/station_data_{}.json".format(station_id)) as ifile:
        json_data = json.load(ifile)
        dataframe = pandas.read_json(json_data)
        # print("Columns of the dataframe: \n{}".format(dataframe.columns))
    return dataframe

"""
    When you run/debug the script from pycharm (or if you run it from your terminal), it starts here !
"""
if __name__ == "__main__":

    # Exemple de dataset univerié
    df = pandas.DataFrame({'value(t)': range(100)})
    df['t+1'] = df['value(t)'].shift(-1)
    df['t+2'] = df['value(t)'].shift(-2)
    df['t+3'] = df['value(t)'].shift(-3)

    ##Execute the evaluate method for station 0 without extracting/using time features
    # je l'ai lancé plusieurs fois pour tous les cas
    mae = evaluate(station_id=2, show_plot=True, use_time_features=False)

