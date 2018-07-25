# Simulated Commute Time Data

This contains a simulation engine for commute time data in a fictional city.

## Insallation

To install in development mode:

```
git clone https://github.com/madrury/commute-times.git
cd commute-times
pip install -e .
```

## Use

To simulate a 5000 record data set:

```
from commute_times.commute_times import CommuteTimeData
commute_data = CommuteTimeData().sample(5000)
```
