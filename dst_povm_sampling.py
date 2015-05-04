"""
.. module:: dst_povm_sampling.py
   :synopsis: Sample projective measurements in the way that DST does
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""
from __future__ import division, absolute_import, print_function, unicode_literals
import numpy as np
from itertools import product

def reseed_choice(a, size=None, replace=True, p=None):
    """Wrapper for the numpy choice function that reseeds before sampling to
    ensure that it doesn't make identical choices accross different parallel
    runs.

    """
    np.random.seed()
    return np.random.choice(a=a, size=size, replace=replace, p=p)

def x_state(anc_outcome, sys_outcome, phi):
    r"""Return the state corresponding to the projective measurement implied by
    a particular outcome (:math:`\pm1`) of the x-measurement on the ancilla and
    a particular outcome (:math:`\widetilde{\pm}1`) of the x-measurement on the
    system:

    .. math::

       \begin{align}
       \vert\psi\rangle&=\cos\frac{\theta}{2}\vert0\rangle+
       \sin\frac{\theta}{2}\vert1\rangle \\
       \theta&=\begin{cases}\operatorname{arctan2}\left(\pm2\cos\varphi,
       \,-\sin^2\varphi\right) & \widetilde{+} \\
       0 & \widetilde{-}\end{cases}
       \end{align}

    :param anc_outcome: :math:`\pm1`, indicates eigenvalue observed on ancilla
                        x-measurement
    :param sys_outcome: :math:`\widetilde{\pm}1`, indicates eigenvalue observed
                        on system x-measurement
    :param phi:         The strength of the interaction
    :returns:           The state represented in the standard computational (z)
                        basis

    """

    theta = np.where(anc_outcome > 0, np.arctan2(2*sys_outcome*np.cos(phi),
                                                 -np.sin(phi)**2), 0)
    return np.array([np.cos(theta/2), np.sin(theta/2)])

def y_state(anc_outcome, sys_outcome, phi):
    r"""Return the state corresponding to the projective measurement implied by
    a particular outcome (:math:`\pm1`) of the y-measurement on the ancilla and
    a particular outcome on the system (:math:`\widetilde{\pm}1`):

    .. math::

       \begin{align}
       \vert\psi\rangle&=\cos\frac{\theta}{2}\vert0\rangle+
       \sin\frac{\theta}{2}\vert1\rangle \\
       \theta&=\operatorname{arccos}\left(\widetilde{\pm}
       \frac{2\left\{\begin{array}{l r}\sin(\varphi+\pi/4) & + \\
       \cos(\varphi+\pi/4) & -\end{array}\right\}^2-1}{2\left\{\begin{array}
       {l r}\sin(\varphi+\pi/4) & + \\ \cos(\varphi+\pi/4) & -\end{array}
       \right\}^2+1}\right)
       \end{align}

    :param anc_outcome: :math:`\pm1`, indicates eigenvalue observed on ancilla
                        z-measurement
    :param sys_outcome: :math:`\widetilde{\pm}1`, indicates eigenvalue observed
                        on system x-measurement
    :param phi:         The strength of the interaction
    :returns:           The state represented in the standard computational (z)
                        basis

    """

    sc = np.where(anc_outcome > 0, np.sin(phi + np.pi/4), np.cos(phi + np.pi/4))
    theta = np.arccos(sys_outcome*(2*sc**2 - 1)/(2*sc**2 + 1))
    return np.array([np.cos(theta/2), np.sin(theta/2)])

def z_state(anc_outcome, phi):
    r"""Return the state corresponding to the projective measurement implied by
    a particular outcome (:math:`\pm1`) of the z-measurement on the ancilla:

    .. math::

       \vert\psi\rangle=\frac{\vert0\rangle+e^{\mp i\varphi}\vert1\rangle}
       {\sqrt{2}}

    :param anc_outcome: :math:`\pm1`, indicates eigenvalue observed on ancilla
                        z-measurement
    :param phi:         The strength of the interaction
    :returns:           The state represented in the standard computational (z)
                        basis

    """

    return np.array([(1. + 0.j)*np.abs(anc_outcome),
                     np.exp(-1.j*anc_outcome*phi)])/np.sqrt(2)

class DSTDistribution(object):
    """A class for sampling from the distribution of projective measurements
    that makes up the POVM for DST

    """
    def __init__(self, phi):
        """Constructor

        :param phi: The strength of the interaction

        """

        # The projective measurement associated with different y-measurement
        # outcomes on the ancilla is chosen with a bias
        y_plus_prob = (2*np.sin(phi + np.pi/2)**2 + 1)/4

        # The probabilities of each of the 6 projective measurements that might
        # be performed
        self.probs = [y_plus_prob/4, y_plus_prob/4, (1 - y_plus_prob)/4,
                      (1 - y_plus_prob)/4, 1/4, 1/4]
        pm = np.array([1, -1])

        # Measurement list (k_ancout_sysout): [y++, y+-, y-+, y--, z+, z-]
        y_measurements = [y_state(ancout, sysout, phi) for ancout, sysout in
                          product(pm, pm)]
        z_measurements = [z_state(ancout, phi) for ancout in pm]
        self.measurements = np.array(y_measurements + z_measurements)

    def sample(self, n=1):
        r"""Get samples from the distribution

        :param n:   The number of samples to draw from the distribution
        :returns:   A ``numpy.array`` of shape (2, n) where the first row
                    contains the :math:`\vert0\rangle` coefficients and the
                    second row contains the :math:`\vert1\rangle` coefficients
                    for the states defining the projective measurements drawn
                    from the distribution

        """

        # Return samples as a (2,n)-shaped array with the first row being the
        # 0-components of the states and the second row being the 1-components
        # of the state
        indices = list(range(len(self.measurements)))
        return self.measurements[reseed_choice(indices, n, p=self.probs)].T

class DSTxyzDistribution(object):
    """A class for sampling from the distribution of projective measurements
    that makes up the POVM for DST when :math:`\sigma_x`, :math:`\sigma_y`, and
    :math:`\sigma_z` are all measured on the meter.

    """
    def __init__(self, phi, anc_probs=[1/4, 1/4, 1/2]):
        """Constructor

        :param phi:         The strength of the interaction
        :param anc_probs:   Probabilities of performing x, y, and z measurements
                            on the ancilla (meter)
        :param p:           List giving the probabilities of performing a
                            :math:`\sigma_x`, :math:`\sigma_y`, or
                            :math:`\sigma_z` measurement on the ancilla (should
                            all be positive, but will automatically be
                            normalized).

        """

        # Normalize the probabilities
        prob_norm = sum(anc_probs)
        n_probs = [prob/prob_norm for prob in anc_probs]

        # The projective measurements associated with different y- and
        # x-measurement outcomes on the ancilla are chosen with biases
        y_plus_prob = (2*np.sin(phi + np.pi/2)**2 + 1)/4
        x_plus_prob = (np.cos(phi)**2 + 1)/2

        # The probabilities of each of the 10 projective measurements that might
        # be performed
        self.probs = [n_probs[0]*x_plus_prob/2, n_probs[0]*x_plus_prob/2,
                      n_probs[0]*(1 - x_plus_prob)/2,
                      n_probs[0]*(1 - x_plus_prob)/2,
                      n_probs[1]*y_plus_prob/2, n_probs[1]*y_plus_prob/2,
                      n_probs[1]*(1 - y_plus_prob)/2,
                      n_probs[1]*(1 - y_plus_prob)/2,
                      n_probs[2]/2, n_probs[2]/2]
        pm = np.array([1, -1])

        # Measurement list (k_ancout_sysout): [y++, y+-, y-+, y--, z+, z-]
        x_measurements = [x_state(ancout, sysout, phi) for ancout, sysout in
                          product(pm, pm)]
        y_measurements = [y_state(ancout, sysout, phi) for ancout, sysout in
                          product(pm, pm)]
        z_measurements = [z_state(ancout, phi) for ancout in pm]
        self.measurements = np.array(x_measurements + y_measurements +
                                     z_measurements)

    def sample(self, n=1):
        r"""Get samples from the distribution

        :param n:   The number of samples to draw from the distribution
        :returns:   A ``numpy.array`` of shape (2, n) where the first row
                    contains the :math:`\vert0\rangle` coefficients and the
                    second row contains the :math:`\vert1\rangle` coefficients
                    for the states defining the projective measurements drawn
                    from the distribution

        """

        # Return samples as a (2,n)-shaped array with the first row being the
        # 0-components of the states and the second row being the 1-components
        # of the state
        indices = list(range(len(self.measurements)))
        return self.measurements[reseed_choice(indices, n, p=self.probs)].T
