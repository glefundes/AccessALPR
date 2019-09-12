import csv
import configparser
import weighted_levenshtein

config = configparser.ConfigParser()

LEV_WEIGHTS = config['DEFAULT']['LevenshteinWeights']
LEV_TRESHOLD = 0.2

with open(LevenshteinWeights, 'r') as readFile:
    csvreader = csv.reader(readFile)
    lines = list(csvreader)[1:]
    for l in lines:
        substitute_costs[ord(l[0]), ord(l[1])] = l[2]


def lev_distance(a, b, weights=LEV_WEIGHTS)
	wlev_dist = weighted_levenshtein.lev(a, b, substitute_costs=weights)
	return wlev_dist