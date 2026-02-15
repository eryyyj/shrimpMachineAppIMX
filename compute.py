AVG_WEIGHT = 0.01
FEED_RATE = 0.06
PROTEIN_RATIO = 0.35

def compute_feed(count):
    biomass = count * AVG_WEIGHT
    feed = biomass * FEED_RATE
    protein = feed * PROTEIN_RATIO
    filler = feed - protein
    return biomass, feed, protein, filler
