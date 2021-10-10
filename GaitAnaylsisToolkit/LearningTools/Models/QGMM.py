import numpy as np
import copy
from GaitAnaylsisToolkit.LearningTools.Models.ModelBase import gaussPDF
from GaitAnaylsisToolkit.LearningTools.Models import ModelBase


class QGMM(ModelBase.ModelBase):
    def __init__(self, nb_states, nb_dim=4, reg=1e-8):
        """
        :param nb_states: number of states
        :param nb_dim: dimensions of the data
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

    def kmeansclustering(self, data, reg=1e-8):
        """
        Perform K-means to init the GMM
        :param data: data to cluster
        :param reg: error term
        :return:
        """

        self.reg = reg
        # Criterion to stop the EM iterative update
        cumdist_threshold = 1e-10
        maxIter = 200
        minIter = 20

        # Initialization of the parameters
        cumdist_old = -1.7977e+308
        nb_step = 0
        self.nbData = data.shape[1]
        id_tmp = np.random.permutation(self.nbData)

        Mu = copy.deepcopy(data[:, id_tmp[:self.nb_states]])
        searching = True
        distTmpTrans = np.zeros((len(data[0]), self.nb_states,))
        idList = []

        while searching:

            # E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
            for i in range(0, self.nb_states):
                # Compute distances
                thing = np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, self.nbData)
                temp = np.power(data - thing, 2.0)
                temp2 = np.sum(temp, 0)
                distTmpTrans[:, i] = temp2

            vTmp = np.min(distTmpTrans, 1)
            cumdist = sum(vTmp)
            idList = []

            for row, min_num in zip(distTmpTrans, vTmp):
                index = np.where(row == min_num)[0]
                idList.append(index[0])

            idList = np.array(idList)
            # M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i in range(self.nb_states):
                # Update the centers
                id = np.where(idList == i)
                Mu[:, i] = np.mean(data[:, id], 2).reshape((1, -1))

            # Stopping criterion %%%%%%%%%%%%%%%%%%%%
            if abs(cumdist - cumdist_old) < cumdist_threshold and nb_step > minIter:
                print('Maximum number of kmeans iterations, ' + str(abs(cumdist - cumdist_old)) + ' is reached')
                print('steps reached, ' + str(nb_step) + ' is reached')
                searching = False

            cumdist_old = cumdist
            nb_step = nb_step + 1

            if nb_step > maxIter:
                print('steps reached, ' + str(nb_step) + ' is reached')
                searching = False
            print("maxitter ", nb_step)

        self.mu = Mu

        return idList

    def em(self, data, reg=1e-8, maxiter=2000):
        """
        Perform the EM algorithm
        :param data: data to learn
        :param reg: error
        :param maxiter:  max number of iterations
        :return:
        """
        self.reg = reg

        nb_min_steps = 50  # min number of iterations
        nb_max_steps = maxiter  # max number of iterations
        nb_samples = data.shape[1]

        data = data.T
        searching = True
        LL = np.zeros(nb_max_steps)
        it = 0
        GAMMA = None
        while searching:

            # E - step
            L = np.zeros((self.nb_states, nb_samples))

            for i in range(self.nb_states):
                L[i, :] = self.priors[i] * gaussPDF(data.T, self.mu[:, i], self.sigma[i])

            GAMMA = L / np.sum(L, axis=0)
            GAMMA2 = GAMMA / np.sum(GAMMA, axis=1)[:, np.newaxis]

            # M-step
            for i in range(self.nb_states):
                # update priors
                self.priors[i] = np.sum(GAMMA[i, :]) / self.nbData
                self.mu[:, i] = data.T.dot(GAMMA2[i, :].reshape((-1, 1))).T
                mu = np.matlib.repmat(self.mu[:, i].reshape((-1, 1)), 1, self.nbData)
                diff = (data.T - mu)
                self.sigma[i] = diff.dot(np.diag(GAMMA2[i, :])).dot(diff.T) + np.eye(self.nb_dim) * self.reg

            # self.priors = np.mean(GAMMA, axis=1)

            LL[it] = np.sum(np.log(np.sum(L, axis=0)))  # / self.nbData
            # Check for convergence
            if it > nb_min_steps:
                if LL[it] - LL[it - 1] < 0.00001 or it == (maxiter - 1):
                    searching = False

            it += 1

        self.BIC = self.BIC_score(LL[it - 1])
        return GAMMA, self.BIC
