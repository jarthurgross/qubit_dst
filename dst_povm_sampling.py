"""
.. module:: dst_povm_sampling.py
   :synopsis: Sample projective measurements in the way that DST does
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""
from __future__ import division, absolute_import, print_function, unicode_literals
import numpy as np
from numpy import sqrt, exp, sin, cos, arccos, pi, arctan2

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

    theta = np.where(sys_outcome > 0, arctan2(2*sys_outcome*cos(phi),
                                              -sin(phi)**2), 0)
    return np.array([cos(theta/2), sin(theta/2)])

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

    sc = np.where(anc_outcome > 0, sin(phi + pi/4), cos(phi + pi/4))
    theta = arccos(sys_outcome*(2*sc**2 - 1)/(2*sc**2 + 1))
    return np.array([cos(theta/2), sin(theta/2)])

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

    return np.array([(1. + 0.j)*np.ones(anc_outcome.shape),
                     exp(-1.j*anc_outcome*phi)])/sqrt(2)

class DSTDistribution(object):
    """A class for sampling from the distribution of projective measurements
    that makes up the POVM for DST

    """
    def __init__(self, phi):
        """Constructor

        :param phi: The strength of the interaction

        """

        self.phi = phi
        self.y_plus_prob = (2*sin(phi + pi/2)**2 + 1)/4
        pm = np.array([1, -1])
        # TODO: Fix these arrays so they can be used instead of calling the
        # functions a bunch of times in the sampling routines.
        sys_outcomes, anc_outcomes = np.meshgrid(pm, pm)
        self.y_states = y_state(anc_outcomes, sys_outcomes, phi)
        self.z_states = z_state(pm, phi)

    def sample(self, n=1):
        r"""Get samples from the distribution

        :param n:   The number of samples to draw from the distribution
        :returns:   A ``numpy.array`` of shape (2, n) where the first row
                    contains the :math:`\vert0\rangle` coefficients and the
                    second row contains the :math:`\vert1\rangle` coefficients
                    for the states defining the projective measurements drawn
                    from the distribution

        """

        pm = np.array([1, -1])
        # We will generate random inputs for the functions, but since the z
        # states have one less free parameter we use a lambda to discard the
        # unused z-eigenvalue
        anc_meas = np.array([y_state, lambda anc_outcome, discard, phi:
                             z_state(anc_outcome, phi)])
        # This array stores the anc_outcome for y or z states
        rand_pm_vals = reseed_choice(pm, n)
        y_anc_outcomes = reseed_choice(pm, n, p=[self.y_plus_prob,
                                                    1 - self.y_plus_prob])
        rand_anc_meas = reseed_choice(anc_meas, n)
        samples = [meas(pm_val, y_anc_outcome, self.phi) for meas, pm_val,
                   y_anc_outcome in zip(rand_anc_meas, rand_pm_vals,
                                        y_anc_outcomes)]

        # Return samples as a (2,n)-shaped array with the first row being the
        # 0-components of the states and the second row being the 1-components
        # of the state
        return np.array(samples).T

class DSTxyzDistribution(object):
    """A class for sampling from the distribution of projective measurements
    that makes up the POVM for DST when :math:`\sigma_x`, :math:`\sigma_y`, and
    :math:`\sigma_z` are all measured on the meter.

    """
    def __init__(self, phi, anc_prob=[1/4, 1/4, 1/2]):
        """Constructor

        :param phi: The strength of the interaction
        :param p:   List giving the probabilities of performing a
                    :math:`\sigma_x`, :math:`\sigma_y`, or :math:`\sigma_z`
                    measurement on the ancilla (should all be positive, but
                    will automatically be normalized).

        """

        self.phi = phi
        self.p = p
        self.x_plus_prob = (cos(phi)**2 + 1)/2
        self.y_plus_prob = (2*sin(phi + pi/2)**2 + 1)/4
        pm = np.array([1, -1])
        # TODO: Fix these arrays so they can be used instead of calling the
        # functions a bunch of times in the sampling routines.
        self.x_states = x_state(np.array([pm, pm]).T, np.array([pm, pm]), phi)
        self.y_states = y_state(np.array([pm, pm]).T, np.array([pm, pm]), phi)
        self.z_states = z_state(pm, phi)

    def sample(self, n=1):
        r"""Get samples from the distribution

        :param n:   The number of samples to draw from the distribution
        :returns:   A ``numpy.array`` of shape (2, n) where the first row
                    contains the :math:`\vert0\rangle` coefficients and the
                    second row contains the :math:`\vert1\rangle` coefficients
                    for the states defining the projective measurements drawn
                    from the distribution

        """

        pm = np.array([1, -1])
        # We will generate random inputs for the functions, but since the z
        # states have one less free parameter we use a lambda to discard the
        # unused z-eigenvalue
        anc_meas = np.array([x_state, y_state,
                             lambda anc_outcome, discard, phi:
                             z_state(anc_outcome, phi)])
        # This array stores the anc_outcome for x, y, or z states
        rand_pm_vals = reseed_choice(pm, n)
        x_anc_outcomes = reseed_choice(pm, n, p=[self.x_plus_prob,
                                                    1 - self.x_plus_prob])
        y_anc_outcomes = reseed_choice(pm, n, p=[self.y_plus_prob,
                                                    1 - self.y_plus_prob])
        # Randomize which measurement is performed on the ancilla (0->x, 1->y,
        # 2->z)
        # TODO: Make this part work
        rand_anc_meas = reseed_choice(anc_meas, n)
        samples = [meas(pm_val, y_anc_outcome, self.phi) for meas, pm_val,
                   y_anc_outcome in zip(rand_anc_meas, rand_pm_vals,
                                        y_anc_outcomes)]

        # Return samples as a (2,n)-shaped array with the first row being the
        # 0-components of the states and the second row being the 1-components
        # of the state
        return np.array(samples).T
