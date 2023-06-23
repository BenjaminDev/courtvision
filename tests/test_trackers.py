from courtvision.trackers import ParticleFilter, StateIdx


def test_tracker_init():
    num_particles = 1000
    tracker = ParticleFilter()
