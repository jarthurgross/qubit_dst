from nose.tools import assert_almost_equal, assert_equal
import numpy as np
import qubit_dst.dst_povm_sampling as samp

def check_state_norms(samples):
    diffs = np.sum(np.abs(samples)**2, axis=0) - np.ones(samples.shape[1])
    assert_almost_equal(max(np.abs(diffs)), 0, 7)

def test_sampling():
    phis = [0, 1, 2]
    for phi in phis:
        dist = samp.DSTDistribution(phi)
        samples = dist.sample(2**8)
        check_state_norms(samples)
