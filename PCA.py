import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, data, n, k):
        self.data = data
        self.original = n  # Initial Dimension of the data
        self.reduced_dimension = k  # Dimension to be reduced

    def train_pca(self):
        n = self.data.shape[0]  # Number of data points
        print(type(self.data))
        print("---------------Inbuilt Covariance--------------")
        cov = np.cov(self.data.transpose())
        print(cov)
        print("-------------------MEAN------------------------")
        mean = np.mean(self.data, axis=0)
        print(mean)
        print("-------------------BEFORE---------------------")
        print(self.data)
        self.data = self.data - mean
        print("-------------------AFTER----------------------")
        print(self.data)
        covariance_matrix = (1/n)*(np.dot(self.data.transpose(), self.data))
        print("------------------COVARIANCE MATRIX-----------")
        print(covariance_matrix)
        print("--------------EIGEN VALUES AND VECTOR---------")
        try:
            eigen_val, eigen_vec = np.linalg.eig(covariance_matrix)
            print("Eigen Values")
            print(eigen_val)
            print("Eigen Vectors")
            print(eigen_vec)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print("Error : Doesn't Converge")
                exit(0)
        # Sort Eigen values and get the Indices
        eigen_val_sorted = np.argsort(eigen_val)
        # m = np.array([103.15887101,  19.19346854,   4.62240609, 20, 1, 9])
        # n = np.argsort(m)
        # print(n)
        # n = n[::-1]
        # print(n)
        print("--------------SORTED EIGEN VALUES-------------")
        print(eigen_val_sorted)
        eigen_val_sorted = eigen_val_sorted[::-1]
        print(eigen_val_sorted)
        # We only need K dimensions, so pick first K Eigen Vectors based on sorted Eigen Values
        print(type(eigen_val_sorted))
        k_sorted_eigen_vals = eigen_val_sorted[:self.reduced_dimension]
        print(k_sorted_eigen_vals)
        # Selecting K Eigen vectors based on Indices
        sorted_eigen_vecs = eigen_vec[:, k_sorted_eigen_vals]
        print(sorted_eigen_vecs)


if __name__ == "__main__":
    # Process the Data
    print("Processing the data")
    dataframe = pd.read_csv('data.txt', names=["x", "y", "z"], delimiter="\t")
    # converting pandas dataframe to numpy array
    dataRV = dataframe.to_numpy()
    obj = PCA(dataRV, 3, 2)
    obj.train_pca()

