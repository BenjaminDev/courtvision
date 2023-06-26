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




Tracking the ball under such dynamics makes for an interesting problem. The glass walls lead to refections and which intern creates "true" false detections. This paper explores the use of a particle filter with internal state of ${x,y,z, dot(x), dot(y), dot(z), dot.double(x), dot.double(y), dot.double(z),dot.triple(x), dot.triple(y), dot.triple(z)}$ to track the balls trajectory. The experimental setup consists of a single stationary camera

= Unwinding events

== Single image camera calibration <single_camera>

Camera calibration is the process of estrimating both the intrinsic and extrinsic camera parameters. The intrinsic parameters $K$ (camera matrix)  and $d$ (distortion cofficients) together with the pinhole camera model determine how rays are projected onto the sensor. The extrinsics $R$ (rotation matrix) and $t$ (translation vector) relates points in the world coordinate system to the camera coordinate system. Without access to the camera, calibration was performed by exploiting known scene geometry. Using the Padel court specifications @court_labelling twenty six point correspondences were identified.

#figure(
  image("images/court_labelling.png", width: 100%),
  caption: [Twenty six points used for camera calibration and camera pose estimation.],
) <court_labelling>


Zhang's calibration approach assumes camera coordinates all lie on the same plane. The padel court scene has many co-planar points. For example: all points on the floor; all points on the front glass; or all points on at a height of the glass are co-planar. In total eight such planes were used:
_floor_plane_,_left_wall_vertical_plane_,_right_wall_vertical_plane_,_front_wall_vertical_plane_,_back_wall_vertical_plane_,_center_vertical_plane_ (net),_top_horizontal_plane_,_topfence_horizontal_plane_.

The manual annotation process included an unquantified amount of noise. To eleaviate  *CourtVisions'* implementation added gaussian noise to each and did an exhausative search for the best $K$ and $d$. The mean reprojection error was used to determine the best camera matrix and distortion coeeficeints. Using this method a reprojection error of $3.50$ pixels in images of resolution $1920 x 1080$.

A similar set of point correspondences were used to estimate the pose of the camera relative the court. Estimating camera pose from known point correspondeces was done in the traditional method using OpenCV's solvepnp implementation. This resulted in a reprojection error of $9.78$ pixels.
TODO: quantify this in meters!!

== Image plane object detection
The task of detecting players and the ball in each frame is a well understood problem. There are a plethora of off-the-shelf models that can be finetuned to perform this task. Howerver, the datasets available in the domain of padel is limited. To this end a cyclic pipeline of train, evealuate, label, train was established. For detecting the players a Yolov8 @Jocher_YOLO_by_Ultralytics_2023 model was used. Ball detection uses a Faster-RCNN @Ren2015 model. All detections were passed to the tracker (see @tracking) which did player re-identification and tracking. The dataset used for fintuning these models is made available @de_Charmoy_CourtVision_2023.


== World coordinate frame object tracking <tracking>
Image plane object detection provides pixel locations of the ball and players per frame. Camera calibrartion gave a mapping from world coordinates to image plane pixel locations. By combining these two results and exploiting the dynamics of a ball in flight estimating the world coordinates of the ball was possible. A particle filter is used to estimate the state of the ball. The state is defined by a set of $N$ particles where each particle $n$ is a vector $arrow(X)#sub[p=n] = {x,y,z, dot(x), dot(y), dot(z), dot.double(x), dot.double(y), dot.double(z), w}$. $x$ is the position $dot(x)$ is the velocity and   $dot.double(x)$ is the acceleration in the $hat(x)$ direction. Similarly for $y$ and $z$. $w$ is the weight of the particle. The state is initialized at random with $x,y,x$ drawn from a uniform distribution over the cube making up the court. The velocity is drawn from a normal distribution with mean $0$ and standard deviation $8.3$. The acceleration is drawn from a normal distribution with mean $0$ and standard deviation $0.1$ with $dot.double(z) = -9.8$. The weight is set to $1/N$. The state is updated every $d\t = 1/f$ seconds where $f$ is the frame rate of the video. Updates propergate each particle according to the following pseudo code:

explain particle filter: https://www.cs.cmu.edu/~15381-s19/recitations/rec12/Recitation_12.pdf

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


 The state is updated using a constant velocity model with a process noise of $0.1$ and a measurement noise of $0.1$. The state is initialized using the first detection of the ball. With each successive frame the state is updated using:

. The state is then projected back onto the image plane and the detection with the lowest euclidean distance is assigned to the ball. This process is repeated for each frame.
