from courtvision.trackers import StateIdx, Tracker


def test_tracker_init():
    num_particles = 1000
    tracker = Tracker()
