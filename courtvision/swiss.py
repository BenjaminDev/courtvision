from pathlib import Path

import numpy as np


def save_camera_params(
    *, file_name, homography, intrinsic_matrix=None, distortion_coeffs=None
):
    """Save camera parameters to a file."""
    file_path = Path(file_name)
    if intrinsic_matrix is not None:
        np.save(file_path.parent / "intrinsic_matrix", intrinsic_matrix)
    if distortion_coeffs is not None:
        np.save(file_path.parent / "distortion_coeffs", distortion_coeffs)
    np.save(file_path.parent / "homography", homography)


# class BYTETracker:

#     def __init__(self, args, frame_rate=30):
#         self.tracked_stracks = []  # type: list[STrack]
#         self.lost_stracks = []  # type: list[STrack]
#         self.removed_stracks = []  # type: list[STrack]

#         self.frame_id = 0
#         self.args = args
#         self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
#         self.kalman_filter = self.get_kalmanfilter()
#         self.reset_id()

#     def update(self, results, img=None):
#         self.frame_id += 1
#         activated_starcks = []
#         refind_stracks = []
#         lost_stracks = []
#         removed_stracks = []

#         scores = results.conf
#         bboxes = results.xyxy
#         # add index
#         bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
#         cls = results.cls

#         remain_inds = scores > self.args.track_high_thresh
#         inds_low = scores > self.args.track_low_thresh
#         inds_high = scores < self.args.track_high_thresh

#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = bboxes[inds_second]
#         dets = bboxes[remain_inds]
#         scores_keep = scores[remain_inds]
#         scores_second = scores[inds_second]
#         cls_keep = cls[remain_inds]
#         cls_second = cls[inds_second]

#         detections = self.init_track(dets, scores_keep, cls_keep, img)
#         """ Add newly detected tracklets to tracked_stracks"""
#         unconfirmed = []
#         tracked_stracks = []  # type: list[STrack]
#         for track in self.tracked_stracks:
#             if not track.is_activated:
#                 unconfirmed.append(track)
#             else:
#                 tracked_stracks.append(track)
#         """ Step 2: First association, with high score detection boxes"""
#         strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
#         # Predict the current location with KF
#         self.multi_predict(strack_pool)
#         if hasattr(self, 'gmc'):
#             warp = self.gmc.apply(img, dets)
#             STrack.multi_gmc(strack_pool, warp)
#             STrack.multi_gmc(unconfirmed, warp)

#         dists = self.get_dists(strack_pool, detections)
#         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

#         for itracked, idet in matches:
#             track = strack_pool[itracked]
#             det = detections[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)
#         """ Step 3: Second association, with low score detection boxes"""
#         # association the untrack to the low score detections
#         detections_second = self.init_track(dets_second, scores_second, cls_second, img)
#         r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
#         # TODO
#         dists = matching.iou_distance(r_tracked_stracks, detections_second)
#         matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
#         for itracked, idet in matches:
#             track = r_tracked_stracks[itracked]
#             det = detections_second[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)

#         for it in u_track:
#             track = r_tracked_stracks[it]
#             if track.state != TrackState.Lost:
#                 track.mark_lost()
#                 lost_stracks.append(track)
#         """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
#         detections = [detections[i] for i in u_detection]
#         dists = self.get_dists(unconfirmed, detections)
#         matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
#         for itracked, idet in matches:
#             unconfirmed[itracked].update(detections[idet], self.frame_id)
#             activated_starcks.append(unconfirmed[itracked])
#         for it in u_unconfirmed:
#             track = unconfirmed[it]
#             track.mark_removed()
#             removed_stracks.append(track)
#         """ Step 4: Init new stracks"""
#         for inew in u_detection:
#             track = detections[inew]
#             if track.score < self.args.new_track_thresh:
#                 continue
#             track.activate(self.kalman_filter, self.frame_id)
#             activated_starcks.append(track)
#         """ Step 5: Update state"""
#         for track in self.lost_stracks:
#             if self.frame_id - track.end_frame > self.max_time_lost:
#                 track.mark_removed()
#                 removed_stracks.append(track)

#         self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
#         self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
#         self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
#         self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
#         self.lost_stracks.extend(lost_stracks)
#         self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
#         self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
#         self.removed_stracks.extend(removed_stracks)
#         if len(self.removed_stracks) > 1000:
#             self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum
#         return np.asarray(
#             [x.tlbr.tolist() + [x.track_id, x.score, x.cls, x.idx] for x in self.tracked_stracks if x.is_activated],
#             dtype=np.float32)

#     def get_kalmanfilter(self):
#         return KalmanFilterXYAH()

#     def init_track(self, dets, scores, cls, img=None):
#         return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

#     def get_dists(self, tracks, detections):
#         dists = matching.iou_distance(tracks, detections)
#         # TODO: mot20
#         # if not self.args.mot20:
#         dists = matching.fuse_score(dists, detections)
#         return dists

#     def multi_predict(self, tracks):
#         STrack.multi_predict(tracks)

#     def reset_id(self):
#         STrack.reset_id()

#     @staticmethod
#     def joint_stracks(tlista, tlistb):
#         exists = {}
#         res = []
#         for t in tlista:
#             exists[t.track_id] = 1
#             res.append(t)
#         for t in tlistb:
#             tid = t.track_id
#             if not exists.get(tid, 0):
#                 exists[tid] = 1
#                 res.append(t)
#         return res

#     @staticmethod
#     def sub_stracks(tlista, tlistb):
#         """ DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
#         stracks = {t.track_id: t for t in tlista}
#         for t in tlistb:
#             tid = t.track_id
#             if stracks.get(tid, 0):
#                 del stracks[tid]
#         return list(stracks.values())
#         """
#         track_ids_b = {t.track_id for t in tlistb}
#         return [t for t in tlista if t.track_id not in track_ids_b]

#     @staticmethod
#     def remove_duplicate_stracks(stracksa, stracksb):
#         pdist = matching.iou_distance(stracksa, stracksb)
#         pairs = np.where(pdist < 0.15)
#         dupa, dupb = [], []
#         for p, q in zip(*pairs):
#             timep = stracksa[p].frame_id - stracksa[p].start_frame
#             timeq = stracksb[q].frame_id - stracksb[q].start_frame
#             if timep > timeq:
#                 dupb.append(q)
#             else:
#                 dupa.append(p)
#         resa = [t for i, t in enumerate(stracksa) if i not in dupa]
#         resb = [t for i, t in enumerate(stracksb) if i not in dupb]
#         return resa, resb
