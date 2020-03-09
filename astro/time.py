import numpy as np


def julian_day(day, month, year):
    j0 = 367 * year - int((7 / 4) * (year + int((month + 9) / 12))) + int(275 * month / 9) + day + 1721013.5
    return j0


def julian_date(day, month, year, hour_utc, min_utc, seconds_utc):
    j0 = julian_day(day, month, year)  # Calculate Julian Date (of whole day)
    ut = hour_utc + (min_utc / 60) + (seconds_utc / 3600)
    jd = j0 + ut / 24
    return jd


def sidereal_time(day, month, year, hour_utc, min_utc, seconds_utc, longitude):
    # longitude is the eastward longitude
    j0 = julian_day(day, month, year)  # Calculate Julian Date (of whole day)
    t0 = (j0 - 2451545.0) / 36525  # Time between J2000 and current julian date in centuries
    sidereal_greenwich = 100.4606184 + 36000.77004 * t0 + 0.000387922 * np.square(t0) - 2.583e-8 * np.power(t0, 3)
    ut = hour_utc + (min_utc / 60) + (seconds_utc / 3600)
    sidereal_greenwich += 360.98564724 * (ut / 24)
    sidereal = sidereal_greenwich + longitude  # Sidereal
    sidereal %= 360
    return sidereal
