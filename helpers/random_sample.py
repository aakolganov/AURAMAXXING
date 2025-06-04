
class RandomSample(dict):
    """A dict subclass that returns random samples from its values on each access."""
    def __init__(self, *args, **kwargs):
        super(RandomSample, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        random_distribution = super(RandomSample, self).__getitem__(key)
        if hasattr(random_distribution, "rvs"):
            return random_distribution.rvs()
        return random_distribution