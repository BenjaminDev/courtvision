#import "template.typ": *
#let bluelink(url, name) = {
  set text(blue)
  [#link(url)[#name]]
}
#let ms2 = {
  [#table("ms")]
}

#show: ieee.with(
  title: "Estimating 3D trajectories from single camera padel videos",
  abstract: [
    Extracting useful information from single viewpoint videos of padel games relies on relating objects within each frame to a world coordinate system. The estimation of ball and player world coordinates using a single camera is an inherently ill posed problem. The measurement space is 2D in pixel units and the state space is 3D in meters. This paper explores the use of prior knowledge about the scene (court geometry) and object dynamics to produce world coordinate estimates. Additionally, a technique for single image camera calibration is presented that exploits the known scene geometry. Evaluation is done on a new dataset of live broadcast footage from the _Ooredoo Qatar Major 2023_ tournament. The dataset as well as code is made available to the community as #bluelink("https://benjamindev.github.io/courtvision/", "CourtVision").



    // The padel court scene has many known geometric features such as court markings, net hight, wall (glass) and fence hights and these are exploited to perform both camera calibration and pose estimation. Robust mappings from 3D world coordinates to 2D image plane coordinates
  ],
  authors: (
    (
      name: "Benjamin de Charmoy",
      department: [Labs],
      organization: [Thinkst Applied Research],
      location: [Cape Town, South Africa],
      email: "benjamin@thinkst.com"
    ),
  ),
  index-terms: ("Padel analytics", "State estimation", "Single image camera calibration", "Tracking"),
  bibliography-file: "export.bib",
)

= Introduction <intro>
Padel is a racket sport played by teams of two in a 10m by 20m court.
The court has glass walls and the ball can be played off and onto the wall. Scoring works similarly to tennis and matches last between one and two hours. During game play the players and a ball move around in the court. Occasionally players move outside the court to bring the ball back into play. These are valid tactics and rallies continue. This movement is captured by a single broadcast camera at approximately thirty frames per second. A single frame is shown in (see @padelcourt). The information captured is rich and dense. It is however not direct measurements of player and ball dynamics. It is the light radiating from the scene that passes through the cameras optics and falls onto the imaging sensor that is captured. The image formation is well understood @Hartley:2003:MVG:861369 and section @single_camera describes this process. This information is further transformed by compression and coding schemes and resides on disk.

Unpacking events that took place during game play and relating them to one another is explored in this paper. "Out of band" information such as the court dimensions, player and ball dynamics are exploited to achieve this.

Given these video recordings can we reconstruct the events that took place with the goal to extract useful player metrics? Player metrics such as: how much time is spent at positions ${x,y,z}#sub[time]$ during a rally? Or how many meters are covered during a game? Or where on the court are most points won/lost from? The hypothesis is these metrics are useful for both players and coaches as they can be used to improve training and game play. A second hypothesis is player and game metrics enhance the viewing experience for spectators. For example, by overlaying player positions on the court during a rally highlighting positional tactics - that would otherwise go unnoticed by enthusiast.

#figure(
  image("images/padelcourt.png", width: 100%),
  caption: [A typical broadcast view of game play.],
) <padelcourt>

To unwind and untangle these events from a recording four tasks were identified: single image camera calibration and camera pose estimation; image plane object detection; world coordinate frame object tracking; and data visualizations.

As part of this project a dataset *CourtVisionPadelDataset* was compiled from videos from the _Ooredoo Qatar Major 2023_ tournament. This dataset consists of TODO:X hours of live broadcast. It's annotated to give timestamps of rallies and bounding boxes of the ball. Accompanying the dataset is Python library *CourtVision* that implements the techniques desribed in this paper. #bluelink("https://benjamindev.github.io/courtvision/","CourtVision") uses _Kornia_ @Riba2019 a _Pytorch_ @pytorch based computer vision library.


= Unwinding events
This section details the four tasks aimed at unwinding events from a video recording. The first two tasks are concerned with mapping from the world coordinate frame to the image plane and locating regions of interest in each frame. The third task of tracking objects in the world coordinate frame exploits a physics model of the ball and players. The final task is concerned with visualizing the data.
== Single image camera calibration <single_camera>

Camera calibration is the process of estimating both the intrinsic and extrinsic camera parameters. The intrinsic parameters $K$ (camera matrix)  and $d$ (distortion coefficients) together with the pinhole camera model determine how rays are projected onto the sensor. The extrinsics $R$ (rotation matrix) and $t$ (translation vector) relates points in the world coordinate system to the camera coordinate system. Traditionally, camera calibration is done using a calibration rig - typically a flat checker board. The rig is placed at varying orientations in the scene and many images are captured. These images are then used to estimate the cameras' intrinsic parameters. Without access to the camera, calibration was performed by exploiting known scene geometry. Using the Padel court specifications @court_dimensions twenty six point correspondences were identified (see @court_labelling).

#figure(
  image("images/court_labelling.png", width: 100%),
  caption: [Twenty six points used for camera calibration and camera pose estimation.],
) <court_labelling>


Zhangs' @Zhang2002 calibration approach assumes camera coordinates all lie on the same plane - a checker board. To utilize this technique sets of co-planar points are needed. The padel court scene has many co-planar points. For example all points on the floor or all points on the front glass. In total eight such planes were used. Since the camera is stationary only a single frame was annotated per clip. The manual annotation process included an unquantified amount of noise. That is the human annotator may have been off by a few pixels. To alleviate this *CourtVisions'* implementation added gaussian noise to each annotation and did an exhaustive search for the best $K$ and $d$. The mean re-projection error was used to determine the best camera matrix and distortion coefficients. Using this method a re-projection error of $3.50[op("pixel")]$ in images of resolution $1920 times 1080$ pixels.

A similar set of point correspondences were used to estimate the pose of the camera relative the court. Estimating camera pose from known point correspondences was done using OpenCV's _solvepnp_ implementation. This resulted in a re-projection error of $9.78[op("pixel")]$. Thus any point in the world coordinate frame with in the court volume re-projected onto the image plane with an error of $9.78[op("pixel")]$.
#figure(
  image("images/forward_projection_error.png", width: 100%),
  caption: [Forward projection error onto the court floor.],
) <forward_projection_error>


To relate this error to the world coordinate system a grid of points on the court floor were projected onto the image plane. Then for each point a cluster of points with uniform noise of $9.78[op("pixel")]$ were projected back to the court floor. The error is the distance between the projected point and the ground truth point. The mean forward projection error is $161[op("mm")]$. The forward projection error is visualized in @forward_projection_error showing larger errors at the far end of the court. This is an encouraging result shows promise for single camera systems.

== Image plane object detection
The task of object detection (players and the ball) in each frame is a well understood problem. There are a plethora of off-the-shelf models that can be fine-tuned to perform this task. However, the datasets available in the domain of padel is limited. To this end a cyclic pipeline of _train_, _evaluate_, _annotate_, _train_ was established. Using LabelStudio TODO:cite labelStudio a web based annotation tool a dataset of 20 images was annotated. The model was fine tuned and used to predict the ball locations on a further 20 images. These were then checked by a human and corrected. The model was retrained on the total 40 images and the process repeated. This was done until a high quality and diverse dataset was created consisting of 600 images.

For detecting the players a Yolov8 @Jocher_YOLO_by_Ultralytics_2023 model was used. Ball detection uses a Faster-RCNN @Ren2015 model. All detections were passed to the tracker (see @tracking) which did player re-identification and tracking. The dataset can code used for fine-tuning these models is made available @de_Charmoy_CourtVision_2023.


== World coordinate frame object tracking <tracking>
Image plane object detection provides pixel locations of the ball and players per frame. Camera calibration gave a mapping from world coordinates to image plane pixel locations. By combining these two results and exploiting the dynamics of a ball in flight estimating the world coordinates of the ball is presented. The player tracking is done using the assumption that they remain on the court floor. This assumption this is not strictly needed it is valid for the majority of the time and gives focus to ball tracking - the harder problem. The ball tracking is done using a particle filter.
\
Related work @LiuPaul which uses a single camera to track 3D trajectories of a shuttle cock during badminton games used a number of domain specific constraints. They seeded the tracker with initial positions and velocity constraints. This work took the view to shift prior knowledge into a distribution and apply a bayesian filtering process. By placing such constraints into prior distributions the proposed tracker aims to easily generalize to other racket sports. The implementation of this bayesian filtering process is realized as a particle filter.

Particle filters @Doucet2011 represent possible states of the ball as particles each with an associated weight. The full state is defined by a set of $N$ particles where each particle $n$ is a vector $arrow(X)#sub[p=n] = {x,y,z, dot(x), dot(y), dot(z), dot.double(x), dot.double(y), dot.double(z), w}$. $x$ is the position $dot(x)$ is the velocity and   $dot.double(x)$ is the acceleration in the $hat(x)$ direction. Similarly for $y, dot(y), dot.double(y)$ and $z, dot(z), dot.double(z)$ in the $hat(y)$ and $hat(z)$ directions. $w$ is the probability mass at that location in the state space. A measurement (detection) is defined as $arrow(Z)#sub[t] = {u,y,l}$ at time $t$ where $u,v$ are the pixel coordinates of the detection center and $l$ is the label which is in the set ${op("ball"), op("playerA1"), op("playerA2"), op("playerB1"), op("playerB2")}$. The particles are initialized at random with $x,y,x$ drawn from a uniform distribution over the cube making up the court.
Ball speeds during a serve are approximately $20 op("km/h")$ to $60 op("km/h")$. Thus, velocity is drawn from a normal distribution with mean $0.0[op("m/s")]$ and standard deviation $8.3[op("m/s")]$. The acceleration component is drawn from a normal distribution with mean $0.0[m/(s#super[2])]$ and standard deviation $0.1$ with $dot.double(z) = -9.8[m/(s#super[2])]$. The weight is set to $1/N$. The state is updated every $d\t = 1/f$ seconds where $f$ is the frame rate of the video.

Particle filters are a sequential Monte Carlo method and have two steps _predict_ and _update_. During the _predict_ phase a new set of particles are sampled from the previous state according to:
#math.equation(block: true, numbering: "(1)", $p(x#sub[t+1]| z#sub[1:t]) = integral p(x#sub[t]|x#sub[t-1]) p(x#sub[t-1]|z#sub[1:t-1])d\x#sub[t-1]$ )
The _predict_ phase incorporates dynamics of the ball $p(x#sub[t]|x#sub[t-1])$ which propergates each particle according to the following pseudo code:

```py
  def predict(state: Tensor, dt: float) -> Tensor:

    noise = torch.randn_like(state) * motion_noise
    # Update position
    state[:, :3] += state[:, 3:6] * dt + noise
    # Update velocity
    state[:, 3:6] += state[:, 6:9] * dt + noise
    # Update acceleration
    state[:, 6:9]
```
During the _update_ phase the particles are weighted according to the likelihood of the measurement $z#sub[t]$ at time $t$. $p(z#sub[t]|x#sub[t])$ is the likelihood of the measurement given the state. The likelihood is calculated weights updated using the following pseudo code:
```py
  def update(state:Tensor, obs: Tensor) -> Tensor:
    # predicted measurement given the state
    pred_obs = state[:,:3]@world_to_cam@cam_to_image
    # Error predicted observation and the measurement
    error = torch.norm(pred_obs - obs, dim=1)
    # The likelihood of the error
    likelihood = torch.exp(
      -0.5 * error ** 2 / measurement_noise
    )
    # Update weight using likelihood
    state[:, -1] *= likelihood
    # Normalize the weights
    state[:, -1] /= state[:, -1].sum()
    # Resample the particles
    state = resample(state)
```
The ball tracker outputs a state estimate for each frame. This is the mean of the particles weighted by their probability mass. The mean state is then projected back onto the image plane. Since the tracker holds $N$ particles the uncertainty of the state can be estimated. The uncertainty is calculated as the standard deviation of the particles weighted by their probability mass.


TODO: add images of tracking

== Data visualizations <visualizations>
Looking at the player positions during a single, and across multiple, rallies during a game is a useful visualization. It gives insight into player tactics and movement. The player positions are projected onto the court floor. Here player The heatmap is generated by summing the time a player is at a position over all frames for a rally in @rally_heatmap and over multiple rallies in @game_heatmap. Velocity heatmaps are generated in a similar fashion. The velocity is calculated as the difference in position per time interval. The figures come from rally _635b_ and game _0000_ of the @de_Charmoy_CourtVision_2023 dataset.

#figure(
  image("images/rally_heatmap.png", width: 100%),
  caption: [Player positions (left) and speeds (right) during a rally.],
) <rally_heatmap>
#figure(
  image("images/game_heatmap.png", width: 100%),
  caption: [Player positions (left) and speeds (right) during a game.],
) <game_heatmap>
The peak
Theses are examples of the visualizations that can be generated from the data. The full set of visualizations are made available @de_Charmoy_CourtVision_2023.

= Conclusion
(inprogress...)

This work tackled a number of common vision problems and applied them to the domain of padel. The results are encouraging and show promise for single camera systems. The single image camera calibration technique using multiple sets of co-planar points is a useful technique when access to the camera is not possible. The ball and player detection models were fine tuned on a small dataset which was released along side this work. The ball and player trackers coupled with the camera calibration and pose estimation enabled world coordinates tracks. These tracks are artifacts of events that took place during game play. The events were unwound and visualized. The visualizations give insight into player tactics and movement.
