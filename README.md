# Welcome to CourtVision
A library and simple application aims to extract useful infromation from videos of court-based[^1] sports. By detecting and tracking players as well as the ball CourtVision captures data at 30 frames per second. By visualizing this high resolution data it is hoped players and coaches will find interesting take-a-ways.

An associated paper with research goals and results is available [here](docs/paper/main.pdf)






[^1]: Currently only padel is supported.

## CLI
Some commonly used scripts are migrated to a cli so they are available when installation is done via `pip`

* `cv-grab-frames` - grabs frames from the dataset specified in `.env`.
