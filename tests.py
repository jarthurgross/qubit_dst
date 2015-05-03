from nose.tools import assert_almost_equal, assert_equal, assert_true
import numpy as np
import qubit_dst.dst_povm_sampling as samp
from itertools import combinations, product

def check_state_norms(samples):
    diffs = np.sum(np.abs(samples)**2, axis=0) - np.ones(samples.shape[1])
    assert_almost_equal(max(np.abs(diffs)), 0, 7)

def check_probabilities(probs, measurements):
    positive = [prob >= 0 for prob in probs]
    assert_true(all(positive))
    assert_almost_equal(sum(probs), 1, 7)
    assert_equal(len(probs), len(measurements))

def check_states_unique(states):

    # Verify that the states generated are unique
    for state1, state2 in combinations(states, 2):
        diff = state1 - state2
        diff_mag = diff.conj().dot(diff)
        assert_true(diff_mag > 1e-7)

def test_sampling():

    phis = [0, 1, 2]

    for phi in phis:
        dists = [samp.DSTDistribution(phi), samp.DSTxyzDistribution(phi),
                 samp.DSTxyzDistribution(phi, [1/3, 1/3, 1/3]),
                 samp.DSTxyzDistribution(phi, [4, 2, 5])]

        for dist in dists:
            check_probabilities(dist.probs, dist.measurements)
            samples = dist.sample(2**8)
            check_state_norms(samples)

def test_uniqueness():
    # Make sure all the states that are being sampled from are unique (only for
    # values of phi that will cause them to be unique), only taking one outcome
    # from the x-states when the ancilla outcome is negative, because the
    # projectors are the same for both system outcomes
    phis = [1, 2]
    pm = [1, -1]
    for phi in phis:
        x_states = [samp.x_state(ancout, sysout, phi) for ancout, sysout in
                    product(pm, pm)]
        y_states = [samp.y_state(ancout, sysout, phi) for ancout, sysout in
                    product(pm, pm)]
        z_states = [samp.z_state(ancout, phi) for ancout in pm]
        check_states_unique(x_states[:-1] + y_states + z_states)
