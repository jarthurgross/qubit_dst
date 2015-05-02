from nose.tools import assert_almost_equal, assert_equal, assert_true
import numpy as np
import qubit_dst.dst_povm_sampling as samp
from itertools import combinations

def check_state_norms(samples):
    diffs = np.sum(np.abs(samples)**2, axis=0) - np.ones(samples.shape[1])
    assert_almost_equal(max(np.abs(diffs)), 0, 7)

def check_states_unique(states):
    # Verify that the states generated are unique
    for state1, state2 in combinations(states, 2):
        diffs = state1.conj().dot(state2)
        assert_true(diffs > 1e-7)

def test_sampling():
    phis = [0, 1, 2]
    for phi in phis:
        dist = samp.DSTDistribution(phi)
        samples = dist.sample(2**8)
        check_state_norms(samples)

def test_uniqueness():
    # Make sure all the states that are being sampled from are unique (except
    # for certain pathological cases)
    phis = [0, 1, 2]
    for phi in phis:
        dist = samp.DSTDistribution(phi)
        check_states_unique(dist.y_states.reshape((4, 2)))
        check_states_unique(dist.z_states)
