"""
.. module:: dst_povm_sampling.py
   :synopsis: Sample projective measurements in the way that DST does
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""
from __future__ import division, absolute_import, print_function, unicode_literals
import numpy as np
from numpy import sqrt, exp, sin, cos, arccos, pi

def z_state(anc_outcome, phi):
    r"""Return the state corresponding to the projective measurement implied by
    a particular outcome (:math:`\pm1`) of the z-measurement on the ancilla:

    .. math::

       \vert\psi\rangle=\frac{\vert0\rangle+e^{\mp i\varphi}\vert1\rangle}
       {\sqrt{2}}

    :param anc_outcomt: :math:`\pm1`, indicates eigenvalue observed on ancilla
                        z-measurement
    :param phi:         The strength of the interaction
    :returns:           The state represented in the standard computational (z)
                        basis

    """

    return sqrt(2)*np.array([1. + 0.j, exp(-1.j*anc_outcome*phi)])

def y_state(anc_outcome, z_eigval, phi):
    r"""Return the state corresponding to the projective measurement implied by
    a particular outcome (:math:`\pm1`) of the y-measurement on the ancilla and
    the z-eigenvalue (:math:`\widetilde{\pm}1` of the system basis element the
    ancilla coupled to:

    .. math::

       \begin{align}
       \vert\psi\rangle&=\cos\theta/2\vert0\rangle+\sin\theta/2\vert1\rangle
       \theta&=\operatorname{arccos}\left(\widetilde{\pm}
       \frac{2\left\{\begin{array}{l r}\sin(\varphi+\pi/4) & + \\
       \cos(\varphi+\pi/4) & -\end{array}\right\}^2-1}{2\left\{\begin{array}
       {l r}\sin(\varphi+\pi/4) & + \\ \cos(\varphi+\pi/4) & -\end{array}
       \right\}}\right)
       \end{align}

    :param anc_outcomt: :math:`\pm1`, indicates eigenvalue observed on ancilla
                        z-measurement
    :param phi:         The strength of the interaction
    :returns:           The state represented in the standard computational (z)
                        basis

    """

    if anc_outcome > 0:
        sc = sin(phi + pi/4)
    else:
        sc = cos(phi + pi/4)
    theta = arccos((2*sc**2 - 1)/(2*sc**2 + 1))
    return np.array([cos(theta/2), sin(theta/2)])
