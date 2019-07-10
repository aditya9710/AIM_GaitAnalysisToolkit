from Exoskeleton import Exoskeleton
from Vicon import Vicon


class Trial(object):

    def __init__(self, exo_file, vicon_file, dt, notes_file=None):
        self._exo_file = exo_file
        self._vicon_file = vicon_file
        self._notes_file = notes_file
        self._dt = dt
        self._exoskeleton = Exoskeleton.Exoskeleton(self._exo_file)
        self._vicon = Vicon.Vicon(self._vicon_file)

    @property
    def dt(self):
        return self._dt

    @property
    def exoskeleton(self):
        return self._exoskeleton

    @property
    def vicon(self):
        return self._vicon

    @dt.setter
    def dt(self, value):
        self._dt = value

    @exoskeleton.setter
    def exoskeleton(self, value):
        self._exoskeleton = value

    @vicon.setter
    def vicon(self, value):
        self._vicon = value
