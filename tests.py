from nose.tools import assert_almost_equal, assert_equal, assert_true
import numpy as np
import qubit_dst.dst_povm_sampling as samp
from itertools import combinations, product

def check_state_norms(samples):
    diffs = np.sum(np.abs(samples)**2, axis=0) - np.ones(samples.shape[1])
    assert_almost_equal(max(np.abs(diffs)), 0, 7)

def check_states_unique(states):
    # Verify that the states generated are unique
    for state1, state2 in combinations(states, 2):
        diff = state1 - state2
        diff_mag = diff.conj().dot(diff)
        assert_true(diff_mag > 1e-7)

def test_sampling():
    phis = [0, 1, 2]
    for phi in phis:
        dist = samp.DSTDistribution(phi)
        assert_almost_equal(sum(dist.probs), 1, 7)
        samples = dist.sample(2**8)
        check_state_norms(samples)

def test_uniqueness():
    # Make sure all the states that are being sampled from are unique (except
    # for certain pathological cases)
    phis = [1, 2]
    pm = [1, -1]
    for phi in phis:
        y_states = [samp.y_state(ancout, sysout, phi) for ancout, sysout in
                    product(pm, pm)]
        z_states = [samp.z_state(ancout, phi) for ancout in pm]
        check_states_unique(y_states + z_states)
