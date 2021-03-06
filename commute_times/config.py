RAW_DISTANCE_SLOPE = 3.0
RAW_DISTANCE_NONLINEARITY_Y_SCALE = 8.0
RAW_DISTANCE_NONLINEARITY_X_SCALE = 2.0
RAW_DISTANCE_NONLINEARITY_POWER = 1.5

TIME_OF_DAY_FACTOR_SCALING = 3.0

MORNING_COMMUTE_BEGIN = 6.0
MORNING_COMMUTE_MID = 8.0
MORNING_COMMUTE_END = 10.0
EVENING_COMMUTE_BEGIN = 16.0
EVENING_COMMUTE_MID = 18.0
EVENING_COMMUTE_END = 21.0

NIGHTIME_TOD_FACTOR = 0.5
MIDDAY_TOD_FACTOR = 0.75
MORNING_PEAK_TOD_FACTOR = 2.0
EVENING_PEAK_TOD_FACTOR = 2.0

COMMUTE_TIME_PROBABILITIES = [0.05, 0.35, 0.25, 0.35]

CONJESTION_FACTOR_MODE = 1.0
CONJESTION_FACTOR_SHAPE = 1000.0
CONJESTION_FACTOR_MINIMUM = 1.0

FIRST_TO_FOURTH_QUAD_FACTOR_MODE = 5.0
FIRST_TO_FOURTH_QUAD_FACTOR_SHAPE = 1000.0
FIRST_TO_FOURTH_QUAD_FACTOR_MINIMUM = 1.0

FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_MODE = 2.0
FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_SHAPE = 1000.0
FIRST_OR_FOURTH_TO_OTHER_QUAD_FACTOR_MINIMUM = 1.0

SAME_QUADRANT_COMMUTE_TYPES = ['CAR', 'BUS', 'BIKE', 'WALK']
SAME_QUADRANT_COMMUTE_TYPE_PROBABILITIES = [0.2, 0.3, 0.3, 0.2] 

FIRST_TO_SECOND_COMMUTE_TYPES = ['CAR', 'BUS', 'TRAIN', 'BIKE']
FIRST_TO_SECOND_COMMUTE_TYPE_PROBABILITIES = [0.2, 0.2, 0.5, 0.1] 

FIRST_TO_THIRD_COMMUTE_TYPES = ['CAR', 'BUS', 'TRAIN', 'BIKE']
FIRST_TO_THIRD_COMMUTE_TYPE_PROBABILITIES = [0.2, 0.2, 0.5, 0.1] 

OTHER_COMMUTE_TYPES = ['CAR', 'BUS', 'BIKE']
OTHER_COMMUTE_TYPE_PROBABILITIES = [0.5, 0.4, 0.1]

CAR_FACTOR = 5.0

BUS_FACTOR_MODE = 15.0 
BUS_FACTOR_SHAPE = 250.0

TRAIN_FACTOR_MODE = 1.0
TRAIN_FACTOR_SHAPE = 250.0

BIKE_FACTOR_MODE = 20.0 
BIKE_FACTOR_SHAPE = 250.0

WALK_FACTOR_MODE = 35.0
WALK_FACTOR_SHAPE = 250.0
