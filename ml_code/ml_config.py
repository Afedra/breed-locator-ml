import os
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn.decomposition import PCA

class MachineLearningConfig():
    def __init__(self):

        self.root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.training_data = pickle.load(open(os.path.join(self.root_directory, "pk_models", 'training_data-13dimHaraFG-PCA.pd.pk'), 'rb'))
        self.breeds = self.training_data.breed

    def read_training_data(self):
        """
        Reads each of the training data, thresholds it and appends it
        to a List that is converted to numpy array

        Parameters:
        -----------
        training_directory: str; of the training directory

        Returns:
        --------
        a tuple containing
        0: 2D numpy array of the training data with its features in 1D
        1: 1D numpy array of the labels (classifications)
        """

        Tdata = self.training_data.drop('breed', 1)
        data = []
        for i in range(Tdata.shape[0]):
            # only use the first 3 components
            print(Tdata.iloc[i])
            data.append([i for i in Tdata.iloc[i, :3]])

        return (np.array(data), np.array(self.breeds))


    def save_model(self, model, foldername):
        """
        saves a model for later re-use without running the training 
        process all over again. Similar to how pickle works

        Parameters:
        -----------
        model: the machine learning model object
        foldername: str; of the folder to save the model
        """
        save_directory = os.path.join(self.root_directory, 'ml_models', foldername)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        joblib.dump(model, os.path.join(save_directory, foldername+'.pkl'))

    def dimension_reduction(self, train_data, number_of_components):
        pca = PCA(number_of_components)
        return pca.fit_transform(train_data)
