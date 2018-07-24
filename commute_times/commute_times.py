import numpy as np

from commute_times.utils import (
    random_uniform_ellipse, line_segment, sample_gamma,
    compute_raw_distance, make_interval,
    in_first_quad, in_fourth_quad, in_lower_half_plane
)
from commute_times.config import (
    MORNING_COMMUTE_BEGIN,
    MORNING_COMMUTE_MID,
    MORNING_COMMUTE_END,
    EVENING_COMMUTE_BEGIN,
    EVENING_COMMUTE_MID,
    EVENING_COMMUTE_END,
    NIGHTIME_TOD_FACTOR,
    MIDDAY_TOD_FACTOR,
    MORNING_PEAK_TOD_FACTOR,
    EVENING_PEAK_TOD_FACTOR,
    COMMUTE_TIME_PROBABILITIES,
    CONJESTION_FACTOR_MODE,
    CONJESTION_FACTOR_SHAPE,
    CONJESTION_FACTOR_MINIMUM,
    FIRST_TO_FOURTH_QUAD_FACTOR_MODE,
    FIRST_TO_FOURTH_QUAD_FACTOR_SHAPE,
    FIRST_TO_FOURTH_QUAD_FACTOR_MINIMUM,
    FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_MODE,
    FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_SHAPE,
    FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_MINIMUM,
)

def sample_time_of_day(n):
    dead_of_night_samples = np.random.uniform(
        EVENING_COMMUTE_END, 24.0 + MORNING_COMMUTE_BEGIN, size=n) % 24.0
    daytime_commute_samples = np.random.triangular(
        left=MORNING_COMMUTE_BEGIN,
        mode=MORNING_COMMUTE_MID,
        right=MORNING_COMMUTE_END,
        size=n)
    midday_samples = np.random.uniform(
        MORNING_COMMUTE_END, EVENING_COMMUTE_BEGIN, size=n)
    evening_commute_samples = np.random.triangular(
        left=EVENING_COMMUTE_BEGIN,
        mode=EVENING_COMMUTE_MID,
        right=EVENING_COMMUTE_END,
        size=n)
    return np.choose(
        np.random.choice(
            len(COMMUTE_TIME_PROBABILITIES),
            size=n, 
            p=COMMUTE_TIME_PROBABILITIES),
        [
            dead_of_night_samples,
            daytime_commute_samples,
            midday_samples,
            evening_commute_samples
        ]
    )

def compute_commute_time(sources, targets, time_of_day):
    assert sources.shape == targets.shape
    assert sources.shape[1] == 2
    assert time_of_day.shape[0] == sources.shape[0]
    N = sources.shape[0]
    raw_distance = compute_raw_distance(sources, targets)
    # Scale is chosen so that the mode of the conjestion_factor is one.
    conjestion_factor = compute_conjestion_factor(N)
    geometry_factor = compute_geometry_factor(sources, targets)
    time_of_day_factor = compute_time_of_day_factor(time_of_day)
    return (
        conjestion_factor 
            * geometry_factor 
            * time_of_day_factor 
            * raw_distance)

def compute_conjestion_factor(n):
    return np.maximum(sample_gamma(
        n=n, 
        mode=CONJESTION_FACTOR_MODE, 
        shape=CONJESTION_FACTOR_SHAPE), 
    CONJESTION_FACTOR_MINIMUM)

def compute_geometry_factor(sources, targets):
    N = sources.shape[0]
    first_to_fourth_quad_factor = np.maximum(
        sample_gamma(
            n=N, 
            mode=FIRST_TO_FOURTH_QUAD_FACTOR_MODE,
            shape=FIRST_TO_FOURTH_QUAD_FACTOR_SHAPE), 
        FIRST_TO_FOURTH_QUAD_FACTOR_MINIMUM)
    first_or_fourth_to_bottom_half_factor = np.maximum(
        sample_gamma(
            n=N, 
            mode=FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_MODE,
            shape=FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_SHAPE),
        FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_MINIMUM)
    cond_list = [
        in_first_quad(sources) & in_fourth_quad(targets),
        in_fourth_quad(sources) & in_first_quad(targets),
        (in_first_quad(sources) | in_fourth_quad(sources)) 
            & in_lower_half_plane(targets),
        (in_first_quad(targets) | in_fourth_quad(targets)) 
            & in_lower_half_plane(sources),
    ]
    choice_list = [
        first_to_fourth_quad_factor,
        first_to_fourth_quad_factor,
        first_or_fourth_to_bottom_half_factor,
        first_or_fourth_to_bottom_half_factor
    ]
    return np.maximum(
        np.select(cond_list, choice_list, default=1.0), 1.0)

def compute_time_of_day_factor(t):
    cond_list = [
        make_interval(t, MORNING_COMMUTE_BEGIN, MORNING_COMMUTE_MID),
        make_interval(t, MORNING_COMMUTE_MID, MORNING_COMMUTE_END),
        make_interval(t, MORNING_COMMUTE_END, EVENING_COMMUTE_BEGIN),
        make_interval(t, EVENING_COMMUTE_BEGIN, EVENING_COMMUTE_MID),
        make_interval(t, EVENING_COMMUTE_MID, EVENING_COMMUTE_END)
    ]
    choice_list = [
        line_segment(t, MORNING_COMMUTE_BEGIN, NIGHTIME_TOD_FACTOR, 
                        MORNING_COMMUTE_MID, MORNING_PEAK_TOD_FACTOR),
        line_segment(t, MORNING_COMMUTE_MID, MORNING_PEAK_TOD_FACTOR, 
                        MORNING_COMMUTE_END, MIDDAY_TOD_FACTOR),
        MIDDAY_TOD_FACTOR,
        line_segment(t, EVENING_COMMUTE_BEGIN, MIDDAY_TOD_FACTOR,
                        EVENING_COMMUTE_MID, EVENING_PEAK_TOD_FACTOR),
        line_segment(t, EVENING_COMMUTE_MID, EVENING_PEAK_TOD_FACTOR,
                        EVENING_COMMUTE_END, NIGHTIME_TOD_FACTOR)
    ]
    return np.select(cond_list, choice_list, default=NIGHTIME_TOD_FACTOR)
