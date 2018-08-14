import numpy as np
import pandas as pd

from commute_times.utils import (
    random_uniform_ellipse, line_segment, sample_gamma,
    compute_raw_distance, make_interval,
    in_first_quad, in_second_quad, in_third_quad, in_fourth_quad,
    in_same_quad, in_lower_half_plane
)
from commute_times.config import (
    RAW_DISTANCE_SLOPE,
    RAW_DISTANCE_NONLINEARITY_X_SCALE,
    RAW_DISTANCE_NONLINEARITY_Y_SCALE,
    RAW_DISTANCE_NONLINEARITY_POWER,
    TIME_OF_DAY_FACTOR_SCALING,
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
    SAME_QUADRANT_COMMUTE_TYPES,
    SAME_QUADRANT_COMMUTE_TYPE_PROBABILITIES,
    FIRST_TO_SECOND_COMMUTE_TYPES,
    FIRST_TO_SECOND_COMMUTE_TYPE_PROBABILITIES,
    FIRST_TO_THIRD_COMMUTE_TYPES,
    FIRST_TO_THIRD_COMMUTE_TYPE_PROBABILITIES,
    OTHER_COMMUTE_TYPES,
    OTHER_COMMUTE_TYPE_PROBABILITIES,
    CAR_FACTOR,
    BUS_FACTOR_MODE,
    BUS_FACTOR_SHAPE,
    TRAIN_FACTOR_MODE,
    TRAIN_FACTOR_SHAPE,
    BIKE_FACTOR_MODE,
    BIKE_FACTOR_SHAPE,
    WALK_FACTOR_MODE,
    WALK_FACTOR_SHAPE,
)



class CommuteTimeData:

    def sample(self, n):
        sources = random_uniform_ellipse(n)
        targets = random_uniform_ellipse(n)
        time_of_day = sample_time_of_day(n)
        time_of_day_ts = convert_to_timestamp(time_of_day)
        commute_type = sample_commute_type(sources, targets)
        commute_time = sample_commute_time(
            sources, targets, time_of_day, commute_type)
        return pd.DataFrame({
            'source_latitude': sources[:, 0],
            'source_longitude': sources[:, 1],
            'destination_latitude': targets[:, 0],
            'destination_longitude': targets[:, 1],
            'time_of_day_ts': time_of_day_ts,
            'commute_type': commute_type,
            'commute_time': commute_time})


def sample_commute_time(sources, targets, time_of_day, commute_type):
    assert sources.shape == targets.shape
    assert sources.shape[1] == 2
    assert time_of_day.shape[0] == sources.shape[0]
    N = sources.shape[0]
    raw_distance = compute_raw_distance(sources, targets)
    geometry_factor = compute_geometry_factor(sources, targets)
    time_of_day_factor = compute_time_of_day_factor(time_of_day)
    commute_type_factor = compute_commute_type_factor(commute_type)
    commute_time_raw = (
        RAW_DISTANCE_SLOPE * raw_distance
            - RAW_DISTANCE_NONLINEARITY_Y_SCALE * (
                2 / (1 + np.exp(RAW_DISTANCE_NONLINEARITY_X_SCALE * raw_distance)
                         ** RAW_DISTANCE_NONLINEARITY_POWER))
            + TIME_OF_DAY_FACTOR_SCALING * time_of_day_factor
            + geometry_factor
            + commute_type_factor)
    return np.maximum(0.0, commute_time_raw)

def sample_time_of_day(n):
    dead_of_night_samples = np.random.uniform(
        EVENING_COMMUTE_END, 24.0 + MORNING_COMMUTE_BEGIN, size=n) % 24.0
    daytime_commute_samples = np.random.triangular(
        left=MORNING_COMMUTE_BEGIN,
        mode=MORNING_COMMUTE_MID,
        right=MORNING_COMMUTE_END,
        size=n)
    midday_samples = np.random.uniform(
        MORNING_COMMUTE_END - 0.5, EVENING_COMMUTE_BEGIN + 0.5, size=n)
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

def convert_to_timestamp(time_of_day, year=2018, month=8, day=12):
    hour_of_day = np.floor(time_of_day)
    minute_of_day = np.floor(60*(time_of_day - hour_of_day))
    time_of_day_ts = pd.to_datetime(pd.DataFrame({
        'year': 2018,
        'month': 8,
        'day': 13,
        'hour': hour_of_day,
        'minute': minute_of_day
    }))
    return time_of_day_ts

def sample_commute_type(sources, targets):
    N = sources.shape[0]
    same_quadrant_commute_types = np.random.choice(
        SAME_QUADRANT_COMMUTE_TYPES,
        size=N,
        p=SAME_QUADRANT_COMMUTE_TYPE_PROBABILITIES)
    first_to_second_commute_types = np.random.choice(
        FIRST_TO_SECOND_COMMUTE_TYPES,
        size=N,
        p=FIRST_TO_SECOND_COMMUTE_TYPE_PROBABILITIES)
    first_to_third_commute_types = np.random.choice(
        FIRST_TO_THIRD_COMMUTE_TYPES,
        size=N,
        p=FIRST_TO_THIRD_COMMUTE_TYPE_PROBABILITIES)
    other_commute_types = np.random.choice(
        OTHER_COMMUTE_TYPES,
        size=N,
        p=OTHER_COMMUTE_TYPE_PROBABILITIES)
    cond_list = [
        in_same_quad(sources, targets),
        in_first_quad(sources) & in_second_quad(targets),
        in_second_quad(sources) & in_first_quad(targets),
        in_first_quad(sources) & in_third_quad(targets),
        in_third_quad(sources) & in_first_quad(targets)
    ]
    choice_list = [
        same_quadrant_commute_types,
        first_to_second_commute_types,
        first_to_second_commute_types,
        first_to_third_commute_types,
        first_to_third_commute_types,
    ]
    return np.select(cond_list, choice_list, default=other_commute_types)

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

def compute_commute_type_factor(commute_type):
    N = commute_type.shape[0]
    car_factor = CAR_FACTOR
    bus_factor = sample_gamma(
        N, mode=BUS_FACTOR_MODE, shape=BUS_FACTOR_SHAPE)
    train_factor = sample_gamma(
        N, mode=TRAIN_FACTOR_MODE, shape=TRAIN_FACTOR_SHAPE)
    bike_factor = sample_gamma(
        N, mode=BIKE_FACTOR_MODE, shape=BIKE_FACTOR_SHAPE)
    walk_factor = sample_gamma(
        N, mode=WALK_FACTOR_MODE, shape=WALK_FACTOR_SHAPE)
    cond_list = [
        commute_type == 'CAR',
        commute_type == 'BUS',
        commute_type == 'TRAIN',
        commute_type == 'BIKE',
        commute_type == 'WALK'
    ]
    choice_list = [
        car_factor,
        bus_factor,
        train_factor,
        bike_factor,
        walk_factor
    ]
    return np.select(cond_list, choice_list, default=1.0)
