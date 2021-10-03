from argparse import ArgumentParser
from collections import Counter, defaultdict
import numpy as np
import os


def parse_args():
    '''
    Parse arguments.
    '''
    x = ArgumentParser()
    x.add_argument('--in_counties', type=str, default='conf/counties.csv')
    x.add_argument('--in_elections', type=str,
                   default='conf/countypres_2000-2020.csv')
    x.add_argument('--out_counties', type=str, default='data/counties.csv')
    x.add_argument('--bad_states', type=str, default='AK,DC')
    x.add_argument('--num_districts', type=int, default=10)
    x.add_argument('--num_iter', type=int, default=10000)
    x.add_argument('--num_samples', type=int, default=1000)
    x.add_argument('--num_dist_samples', type=int, default=1000)
    return x.parse_args()


def str_to_coord(s, a, z):
    '''
    Parse coordinate from str.
    '''
    assert s
    assert s[-1] == '°'
    if s[0] == '+':
        r = float(s[1:-1])
    elif s[0] == '–':
        r = -float(s[1:-1])
    else:
        assert False
    assert a < r < z
    return r


def str_to_lat(s):
    '''
    Parse latitude from str.
    '''
    return str_to_coord(s, -90, 90)


def str_to_lon(s):
    '''
    Parse longitude from str.
    '''
    return str_to_coord(s, -180, 180)


def str_to_fips(s):
    '''
    Parse FIPS code from str.
    '''
    if s == 'NA':
        return 0
    else:
        return int(s)


def state_str_to_county(state, s):
    '''
    Normalize county name.
    '''
    if state == 'HI':
        for c in "'ʻ":
            s = s.replace(c, '')
    if s == 'LASALLE':
        s = 'LA SALLE'
    s = s.replace('.', '')
    return s


def str_to_vote(s):
    '''
    Parse vote count from str.
    '''
    if s == 'NA':
        return 0
    else:
        return int(s)


def load_counties(f, bad_states):
    '''
    Load counties data from CSV.

    Columns:
        0.
        1.  State (2-letter code)
        2.  County FIPS
        3.  County name
        4.  County seat name
        5.
        6.
        7.
        8.
        9.
        10.
        11.
        12. Latitude (degrees)
        13. Longitude (degrees)

    Example:
        0.  2880
        1.  VA
        2.  51121
        3.  Montgomery
        4.  Christiansburg
        5.  94,392
        6.  1,002.361
        7.  387.014
        8.  5.830
        9.  2.251
        10. 1,008.191
        11. 389.265
        12. +37.174885°
        13. –80.387314°
    '''
    rr = []
    s = open(f).read().strip()
    for i in range(15):
        s = s.replace('[%d]' % i, '')
    for line in s.split('\n'):
        ss = list(map(lambda s: s.strip(), line.split('\t')))
        assert len(ss) == 14

        state = ss[1]
        if state in bad_states:
            continue

        fips = str_to_fips(ss[2])
        lat = str_to_lat(ss[12])
        lon = str_to_lon(ss[13])
        r = state, fips, lat, lon
        rr.append(r)
    return rr


def load_elections(f, bad_states):
    '''
    Load elections data from CSV.

    Columns:
        0.  Year
        1.  State
        2.  State (2-letter code)
        3.  County name
        4.  County FIPS
        5.  Office
        6.  Candidate
        7.  Party
        8.  Candidate Votes
        9.  Total votes
        10. Version
        11. Mode
    
    Example:
        0.  2020
        1.  "VIRGINIA"
        2.  "VA"
        3.  "MONTGOMERY"
        4.  "51121"
        5.  "US PRESIDENT"
        6.  "JOSEPH R BIDEN JR"
        7.  "DEMOCRAT"
        8.  18499
        9.  45037
        10. 20210622
        11. "ABSENTEE"
    '''
    rr = []
    f = open(f)
    next(f)
    for s in f:
        s = s.strip()
        s = s.replace('"', '')
        ss = s.split(',')

        year = int(ss[0])
        if year < 2020:
            continue

        state = ss[2]
        if state in bad_states:
            continue

        fips = str_to_fips(ss[4])
        party = ss[7]
        vote = str_to_vote(ss[8])
        r = state, fips, party, vote
        rr.append(r)
    return rr


def main(args):
    '''
    Load data, clean and merge, save data.
    '''
    counties = load_counties(args.in_counties, args.bad_states)
    states, fipss, lats, lons = zip(*counties)
    assert fipss == tuple(sorted(fipss))
    fips2state = dict(zip(fipss, states))
    fips2lat_lon = dict(zip(fipss, zip(lats, lons)))
    state2fipss = defaultdict(list)
    for state, fips in zip(states, fipss):
        state2fipss[state].append(fips)

    elections = load_elections(args.in_elections, args.bad_states)
    fips2party2vote = defaultdict(Counter)
    for state, fips, party, vote in elections:
        st = fips2state.get(fips)
        if st is None:
            print('Missing:', state, fips, party, vote, '->', st)
            continue
        assert st == state
        fips2party2vote[fips][party] += vote

    d = os.path.dirname(args.out_counties)
    if not os.path.exists(d):
        os.makedirs(d)

    with open(args.out_counties, 'w') as out:
        ss = 'State', 'FIPS', 'Lat', 'Lon', 'Dem', 'Rep'
        line = ','.join(ss) + '\n'
        out.write(line)
        for state in sorted(state2fipss):
            for fips in state2fipss[state]:
                lat, lon = fips2lat_lon.get(fips)
                party2vote = fips2party2vote[fips]
                dem = party2vote['DEMOCRAT']
                rep = party2vote['REPUBLICAN']
                if not dem + rep:
                    continue
                fips = str(fips)
                lat = '%.6f' % lat
                lon = '%.6f' % lon
                dem = str(dem)
                rep = str(rep)
                ss = state, fips, lat, lon, dem, rep
                line = ','.join(ss) + '\n'
                out.write(line)


if __name__ == '__main__':
    main(parse_args())
