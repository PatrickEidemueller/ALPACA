"""
This file contains the implementation of the Monte Carlo dropout networks used in active_learning_dropout.py

There is one implementation which uses MC dropout with a fixed, hand picked dropout ratio
and one implementation that uses concrete_dropout to automatically determine the dropout ration.
"""

#libarys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
# pip install concretedropout 
from concretedropout.tensorflow import ConcreteDenseDropout,\
    get_weight_regularizer, get_dropout_regularizer
from keras.models import Sequential
import tensorflow as tf
from keras import layers
from keras.layers import Dense
import matplotlib.pyplot as plt
import keras.backend as K

# Function for Monte Carlo Dropout Neural Network. The dropout_rate has to be
# tuned manually or with Grid Search. The function returns a fitted model on
# the training data and  the training history
def fit_MC_Dropout_model(X_train, y_train, X_test, y_test, epoch, dropout_rate,
                         learning_rate=0.001):
    
     input_dim = X_train.shape[1]
     
     #create the neural network
     model = Sequential() 
     model.add(layers.Dropout(dropout_rate))
     model.add(Dense(120, input_shape=(input_dim, ), activation='relu')) 
     model.add(layers.Dropout(dropout_rate))
     model.add(Dense(100, activation='relu'))         
     model.add(layers.Dropout(dropout_rate))
     model.add(Dense(100, activation='relu'))   
     model.add(layers.Dropout(dropout_rate))
     model.add(Dense(1, activation='linear'))
     
     # compile and fit the model on training data
     opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
     model.compile(loss='mean_squared_error', optimizer=opt,
                   metrics=['RootMeanSquaredError'])
     history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=epoch, verbose=1)
     
     return model, history

# Function for Concrete Dropout Neural Network. The function returns a fitted 
# model on the training data and  the training history
def fit_concrete_dropout_model(X_train, y_train, X_test, y_test, epoch,
                               learning_rate=0.001):

    n_trainrows, input_dim = X_train.shape
    
    # compute the regularisation values
    wr = get_weight_regularizer(n_trainrows, l=1e-2, tau=1.)
    dr = get_dropout_regularizer(n_trainrows, tau=1.0,
                                 cross_entropy_loss=False)
    
    # create the neural network
    inputs = tf.keras.layers.Input(input_dim, name="inputs")

    dense1 = layers.Dense(120, name="HL1")
    x = ConcreteDenseDropout(dense1, is_mc_dropout=True, weight_regularizer=wr,
                             dropout_regularizer=dr)(inputs)
    x = layers.Activation("relu")(x)
    
    dense2 = layers.Dense(100)
    x = ConcreteDenseDropout(dense2, is_mc_dropout=True, weight_regularizer=wr,
                             dropout_regularizer=dr, name='HL2')(x)
    x = layers.Activation("relu")(x)

    dense3 = layers.Dense(100)
    x = ConcreteDenseDropout(dense3, is_mc_dropout=True, weight_regularizer=wr,
                             dropout_regularizer=dr, name='HL3')(x)
    x = layers.Activation("relu")(x)

    dense4 = layers.Dense(1)
    x = ConcreteDenseDropout(dense4, is_mc_dropout=True, weight_regularizer=wr,
                             dropout_regularizer=dr, name='outputs')(x)
    output = layers.Activation("linear")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output,
                           name="regression_model")
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['RootMeanSquaredError'])
    # print(model)
    history =  model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=epoch, verbose=0)

    return model, history

# Function Neural Network without Regularization. The function returns a fitted 
# model on the training data and  the training history
def fit_model_without_regulization(X_train,y_train, X_test, y_test, epoch,
                                   learning_rate=0.001):
    
     input_dim = X_train.shape[1]
     
     #create the neural network
     model = Sequential() 
     model.add(Dense(120, input_shape=(input_dim, ), activation='relu')) 
     model.add(Dense(100, activation='relu'))         
     model.add(Dense(100, activation='relu'))   
     model.add(Dense(1, activation='linear'))
     
     # compile and fit the model on training data
     opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
     model.compile(loss='mean_squared_error', optimizer=opt,
                   metrics=['RootMeanSquaredError'])
     history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=epoch, verbose=0)
     
     return model, history

def main():
    
    # load Featurevectors for ESOL Dataset
    path_esol_feature_vectors = 'C:/Users/rehan/OneDrive/Master Data Science/\
03_PROJEKT/datasets_darmstadt/featurevector.csv'
    data_raw = pd.read_csv(path_esol_feature_vectors)
    
    # normalize Featurevectors
    scaler = StandardScaler().fit(data_raw.drop(['y'], axis=1))
    X = pd.DataFrame(scaler.transform(data_raw.drop(['y'], axis=1)))
    X = np.array(X)
    y = np.array(data_raw.y)
    
    # split data in train(70%) ,test(30%)        
    test_size = 0.3  
    seed = 8451                                   
    X_test, X_train, y_test, y_train = train_test_split(X, y,             
            train_size=int(X.shape[0] * test_size), random_state=seed)  
    
    # making some definitions for the following for-loop
    rmse_df = pd.DataFrame()
    counter = 0
    runs = 20
    epoch = 50
    dropout_rate = 0.045    ## this was determined by GridSarch by hand
    
    # in this loop each network will be trained 'runs' - times and the RMSE
    # for each training and network is collected in rmse_df
    for i in range(runs):
    
        model, history = fit_concrete_dropout_model(X_train, y_train, X_test,
                                                          y_test, epoch) 
        
        model2, history2 = fit_MC_Dropout_model(X_train, y_train, X_test,
                                      y_test, epoch, dropout_rate=dropout_rate)
    
        model3, history3 = fit_model_without_regulization(X_train ,y_train,
                                                         X_test, y_test, epoch)

        # collecting data for next plot
        rmse_df.insert(loc=counter,column=str(counter), 
            value=[round(history.history['val_root_mean_squared_error'][-1],4),
                   round(history.history['root_mean_squared_error'][-1],4),
                   'Concrete-DO'])
        counter += 1
        
        rmse_df.insert(loc=counter,column=str(counter),
           value=[round(history2.history['val_root_mean_squared_error'][-1],4),
                  round(history2.history['root_mean_squared_error'][-1],4),
                  'MonteCarlo-DO'])
        counter += 1
        
        rmse_df.insert(loc=counter,column=str(counter),
           value=[round(history3.history['val_root_mean_squared_error'][-1],4),
                  round(history3.history['root_mean_squared_error'][-1],4),
                  'No Regularization'])
        counter += 1
    
    # Create Boxplot
    rmse_df = rmse_df.T
    rmse_df.columns = ['Test', 'Training', 'Neural Network with']
    rmse_df[["Test", "Training"]] = rmse_df[["Test", "Training"]].\
        apply(pd.to_numeric)
    rmse_df.boxplot(by='Neural Network with', figsize=(16,8))
    plt.title('Training (epochs: {}) | {} runs'.format(epoch,runs))
    plt.suptitle('') 
    plt.savefig('RMSE von verschiedenen Neuronalen Netzen mit {} runs.png'.format(runs))
    
    # Print Dropout rates
    print('MC-DO Dropoutrate:\n {}'.format(dropout_rate))
    ps = np.array([K.eval(layer.p_logit) for layer in model.layers if 
                   hasattr(layer, 'p_logit')])
    droput_val = tf.nn.sigmoid(ps).numpy()
    print('Concrete-DO Dropoutrate:\n {}'.format(droput_val))

if __name__ == "__main__":
    main()

