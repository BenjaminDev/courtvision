#import "template.typ": *
#show: ieee.with(
  title: "Estimating 3D ball trajectories from single camera padel videos",
  abstract: [
    Extracting useful information from videos of padel games
    The estimnation of objects world coordinates using a single camera is a inherently ill posed problem. The measurement space 2D in pixel units and the state space is 3D in meters. This paper explores the use of prior knwoledge about the scene and object dynamics to produce estimates. The padel court scene has many known geometric features such as court markings, net hight, wall (glass) and fence hights and these are exploited to perform both camera calibration and pose estimation. Robust mappings from 3D world coordinates to 2D image plane coordinates
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
Padel is a racquet sport played by teams of two in a 10m by 20m court.The court has glass walls and the ball can be played off and onto the wall. Scoring works similarly to tennis and matches last between one and two hours. During this time the players and a ball move around in the court  see @padelcourt. This movement is captured by a single broadcast camera at approximatly thirty frames per second. This work setout to unwind those events. Given these video recordings  can we reconstruct the events that took place with the goal to extract useful player metrics.

#figure(
  image("images/padelcourt.png", width: 100%),
  caption: [A typical broadcast view of game play.],
) <padelcourt>

To achieve this 4 tasks were identified: single image camera calibration and camera pose estimation; image plane object detection; world coordinate frame object tracking and data visualizations.

As part of this project a dataset *CourtVisionPadelDataset* was compiled from videos from the _Ooredoo Qatar Major 2023_ tornament. This dataset consistes of TODO:X hours of live broadcast. It's annotated to give timestamps of rallies and bounding boxes of the ball. Accompanying the dataset is Python library *CourtVision* that implements the techniques desribed in this paper. #link("https://benjamindev.github.io/courtvision/")[*CourtVision*] uses _Kornia_ @Riba2019 a _Pytorch_ @pytorch based computer vision library.

== Paper overview <overview>
TODO:
= Unwinding events

== Single image camera calibration <single_camera>

Camera calibration is the process of estrimating both the intrinsic and extrinsic camera parameters. The intrinsic parameters $K$ (camera matrix)  and $d$ (distortion cofficients) together with the pinhole camera model determine how rays are projected onto the sensor. The extrinsics $R$ (rotation matrix) and $t$ (translation vector) relates points in the world coordinate system to the camera coordinate system. Without access to the camera, calibration was performed by exploiting known scene geometry. Using the Padel court specifications @court_labelling twenty six point correspondences were identified.

#figure(
  image("images/court_labelling.png", width: 100%),
  caption: [Twenty six points used for camera calibration and camera pose estimation.],
) <court_labelling>


Zhang's calibration approach assumes camera coordinates all lie on the same plane. The padel court scene has many co-planar points. For example all points on the floor or all points on the front glass. In total eight such planes were used.

The manual annotation process included an unquantified amount of noise. To eleaviate  *CourtVisions'* implementation added gaussian noise to each and did an exhausative search for the best $K$ and $d$. The mean reprojection error was used to determine the best camera matrix and distortion coeeficeints. Using this method a reprojection error of $3.50$ pixels in images of resolution $1920 x 1080$.

A similar set of point correspondences were used to estimate the pose of the camera relative the court. Estimating camera pose from known point correspondeces was done in the traditional method using OpenCV's solvepnp implementation. This resulted in a reprojection error of $9.78$ pixels.
TODO: quantify this in meters!!

== Image plane object detection
The task of detecting players and the ball in each frame is a well understood problem. There are a plethora of off-the-shelf models that can be finetuned to perform this task. Howerver, the datasets available in the domain of padel is limited. To this end a cyclic pipeline of train, evealuate, label, train was established. For detecting the players a Yolov8 @Jocher_YOLO_by_Ultralytics_2023 model was used. Ball detection uses a Faster-RCNN @Ren2015 model. All detections were passed to the tracker (see @tracking) which did player re-identification and tracking. The dataset used for fintuning these models is made available @de_Charmoy_CourtVision_2023.


== World coordinate frame object tracking <tracking>
Image plane object detection provides pixel locations of the ball and players per frame. Camera calibrartion gave a mapping from world coordinates to image plane pixel locations. By combining these two results and exploiting the dynamics of a ball in flight estimating the world coordinates of the ball is possible. The player tracking is done using the assumption that they remain on the court floor. This assumption is valid for the majority of the time. The ball tracking is done using a particle filter.
\
Particle filters @Doucet2011 represent possible states of the ball as particles each with an associated weight. The full state is defined by a set of $N$ particles where each particle $n$ is a vector $arrow(X)#sub[p=n] = {x,y,z, dot(x), dot(y), dot(z), dot.double(x), dot.double(y), dot.double(z), w}$. $x$ is the position $dot(x)$ is the velocity and   $dot.double(x)$ is the acceleration in the $hat(x)$ direction. Similarly for $y$ and $z$. $w$ is the probability mass at that location in the state space. The particles are initialized at random with $x,y,x$ drawn from a uniform distribution over the cube making up the court.
Ball speeds are approximately $20 op("km/h")$ to $60 op("km/h")$ during a serve in padel. Thus, velocity is drawn from a normal distribution with mean $0$ and standard deviation $8.3 [m slash s]$.
The acceleration is drawn from a normal distribution with mean $0$ and standard deviation $0.1$ with $dot.double(z) = -9.8$. The weight is set to $1/N$. The state is updated every $d\t = 1/f$ seconds where $f$ is the frame rate of the video.

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
  def update(state:Tensor, measurement: Tensor) -> Tensor:
    # predicted measurement given the state
    pred_obs = state[:,:3]@world_to_cam@cam_to_image
    # Error predicted observation and the measurement
    error = torch.norm(pred_obs - measurement, dim=1)
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

== Data visualizations <visualizations>
