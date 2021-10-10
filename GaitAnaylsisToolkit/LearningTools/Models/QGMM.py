import numpy as np
from GaitAnaylsisToolkit.LearningTools.Models import ModelBase


class QGMM(ModelBase.ModelBase):
    def __init__(self, nb_states, nb_dim=3, reg=[1e-8]):
        """
        :param nb_states: number of states
        :param nb_dim: dimension of the data
        :param reg: regularization term
        """

        super(QGMM, self).__init__(nb_states, nb_dim, reg)
        self.frames = 1

    def init_params(self, data):
        """
        Sets up all the parameters
        :param data: vector of the data
        :return:
        """

        idList = self.kmeansclustering(data)
        priors = np.ones(self.nb_states) / self.nb_states
        self.sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])
        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        for i in range(self.nb_states):

            idtmp = np.where(idList == i)
            mat = np.vstack((data[:, idtmp][0][0], data[:, idtmp][1][0]))

            for j in range(2, len(data[:, idtmp])):
                mat = np.vstack((mat, data[:, idtmp][j][0]))

            mat = np.concatenate((mat, mat), axis=1)
            priors[i] = len(idtmp[0])
            self.sigma[i] = np.cov(mat) + np.diag(self.reg)

        self.priors = priors / np.sum(priors)

    def train(self, data, maxiter=2000):
        """
        Train the model on the data
        :param data:  data to train on
        :param maxiter: maximum number of iterations
        :return:
        """

        self.init_params(data)
        gamma, BIC = self.em(data, maxiter)
        return gamma, BIC

    def get_model(self):
        """
        Get all the model parameters

        :returns: Model parameters
            - sigma - list covariance matrix
            - mu - list of means
            - priors - list of priors
        """

        return self.sigma, self.mu, self.priors