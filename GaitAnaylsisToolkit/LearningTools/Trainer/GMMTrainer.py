from GaitAnaylsisToolkit.LearningTools.Trainer import TrainerBase
from GaitCore.Core import utilities as utl
from GaitAnaylsisToolkit.LearningTools.Models import GMM, GMR
import numpy as np
from numpy import matlib


class GMMTrainer(TrainerBase.TrainerBase):

    def __init__(self, demo, file_name, n_rf, dt=0.01, reg=[1e-5], poly_degree=[15]):
        """
           :param file_names: file to save training too
           :param n_rfs: number of DMPs
           :param dt: time step
           :return: None
       """

        if len(reg) == len(demo):
            my_reg = [1e-8] + reg
        else:
            my_reg = reg * (1 + len(demo))

        self._kp = 50.0
        self._kv = (2.0 * self._kp) ** 0.5
        rescaled = []
        self.dtw_data = []

        demo_, dtw_data_ = self.resample(demo, 20)
        rescaled = demo_
        self.dtw_data.append(dtw_data_)
        super(GMMTrainer, self).__init__(rescaled, file_name, n_rf, dt, my_reg)

    def train(self, save=True):
        """

        """
        nb_dim = len(self._demo)
        self.gmm = GMM.GMM(nb_states=self._n_rfs, nb_dim=nb_dim)
        tau, motion, sIn = self.gen_path(self._demo)
        gammam, BIC = self.gmm.train(tau)
        sigma, mu, priors = self.gmm.get_model()
        gmr = GMR.GMR(mu=mu, sigma=sigma, priors=priors)
        expData, expSigma, H = gmr.train(sIn, [0], [1], self.reg)   # train the model

        self.data["BIC"] = BIC
        self.data["len"] = len(sIn)
        self.data["H"] = H
        self.data["motion"] = motion
        self.data["expData"] = expData
        self.data["expSigma"] = expSigma
        self.data["sIn"] = sIn
        self.data["tau"] = tau
        self.data["mu"] = mu
        self.data["sigma"] = sigma
        self.data["dt"] = self._dt
        self.data["demos"] = self._demo
        self.data["start"] = self._demo[0][0]
        self.data["goal"] = self._demo[0][-1]
        self.data["dtw"] = self.dtw_data

        if save:
            self.save()
        return self.data

    def gen_path(self, demos):
        """

        """

        self.nbData = len(demos[0])
        self.samples = len(demos)

        alpha = 1.0
        x_ = None
        dx_ = None
        ddx_ = None
        sIn = []
        taux = []

        sIn.append(1.0)  # Initialization of decay term
        for t in range(1, self.nbData):
            sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * self._dt)  # Update of decay term (ds/dt=-alpha s) )

        goal = demos[0][-1]

        for n in range(self.samples):
            demo = demos[n]
            size = demo.shape[0]
            x = utl.spline(np.arange(1, size + 1), demo, np.linspace(1, size, self.nbData))
            dx = np.divide(np.diff(x, 1), np.power(self._dt, 1))
            dx = np.append([0.0], dx[0])
            ddx = np.divide(np.diff(x, 2), np.power(self._dt, 2))
            ddx = np.append([0.0, 0.0], ddx[0])
            goals = np.matlib.repmat(goal, self.nbData, 1)
            tau_ = ddx - (self._kp * (goals.transpose() - x)) / sIn + (self._kv * dx) / sIn

            if x_ is not None:
                x_ = x_ + x.tolist()
                dx_ = np.vstack((dx_, dx))
                ddx_ = np.vstack((ddx_, ddx))
            else:
                x_ = x.tolist()
                dx_ = dx.tolist()
                ddx_ = ddx.tolist()

            taux = taux + tau_[0].tolist()

        t = sIn * self.samples
        tau = np.vstack((t, taux))
        motion = np.vstack((x_, dx_, ddx_))

        return tau, motion, sIn

    def solve_riccati(self, expSigma):
        Ad = np.eye(2)
        Q = np.zeros((2, 2))
        Bd = np.array([[0], [self._dt]])
        Ad[0, 1] = self._dt
        R = np.eye(1) * self.gmm.reg
        P = [np.zeros((2, 2))] * len(expSigma)
        P[-1][0, 0] = np.linalg.pinv(expSigma[-1])

        for ii in xrange(len(expSigma) - 2, -1, -1):
            Q[0, 0] = np.linalg.pinv(expSigma[ii])
            B = P[ii + 1] * Bd
            C = np.linalg.pinv(np.dot(Bd.T * P[ii + 1], Bd) + R)
            D = Bd.T * P[ii + 1]
            F = np.dot(np.dot(Ad.T, B * C * D - P[ii + 1]), Ad)
            P[ii] = Q - F

        self.data["Ad"] = Ad
        self.data["Bd"] = Bd
        self.data["R"] = R
        self.data["P"] = P
