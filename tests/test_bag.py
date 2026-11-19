import numpy as np

from fosconet.bag import quat_to_mat, slerp


def test_quaternion_and_slerp():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])       # 90 deg about z
    R = quat_to_mat(slerp(q0, q1, np.array(0.5)))
    assert np.allclose(R @ [1, 0, 0], [np.cos(np.pi / 4), np.sin(np.pi / 4), 0], atol=1e-9)
    assert np.allclose(quat_to_mat(q1) @ [1, 0, 0], [0, 1, 0], atol=1e-9)
