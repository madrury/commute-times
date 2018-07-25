import numpy as np


def random_uniform_disk(n, radius=1.0):
    theta = np.random.uniform(low=0.0, high=2*np.pi, size=n)
    r = np.sqrt(np.random.uniform(low=0.0, high=radius**2, size=n))
    x, y = r * np.cos(theta), r * np.sin(theta)
    return np.column_stack((x, y))

def random_uniform_ellipse(n, M=np.eye(2)):
    return random_uniform_disk(n) @ M

def line_segment(x, x0, y0, x1, y1):
    m = (y1 - y0) / (x1 - x0)
    return m * (x - x0) + y0

def sample_gamma(n, mode=1.0, shape=5.0):
    scale = mode / (shape - 1)
    return np.random.gamma(shape=shape, scale=scale, size=n)

def compute_raw_distance(sources, targets):
    raw_distance = (
        np.abs(sources[:, 0] - targets[:, 0]) +
        np.abs(sources[:, 1] - targets[:, 1]))
    return raw_distance

def make_interval(t, begin, end):
    return (t >= begin) & (t <= end)

def in_first_quad(xy):
    return (xy[:, 0] >= 0) & (xy[:, 1] >= 0)

def in_second_quad(xy):
    return (xy[:, 0] >= 0) & (xy[:, 1] <= 0)

def in_third_quad(xy):
    return (xy[:, 0] <= 0) & (xy[:, 1] <= 0)

def in_fourth_quad(xy):
    return (xy[:, 0] <= 0) & (xy[:, 1] >= 0)

def in_same_quad(xy0, xy1):
    return (
        (in_first_quad(xy0) & in_first_quad(xy1))
        | (in_second_quad(xy0) & in_second_quad(xy1))
        | (in_third_quad(xy0) & in_third_quad(xy1))
        | (in_fourth_quad(xy0) & in_fourth_quad(xy1)))

def in_lower_half_plane(xy):
    return in_second_quad(xy) | in_third_quad(xy)
