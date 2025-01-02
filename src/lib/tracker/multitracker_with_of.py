import itertools
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process

from tracker import matching

from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker_OF(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0, flow=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        # 追加-----------------------------------------------------------------------------------------

        # detectionsを補正（最終的にupdate_detections_with_optical_flowメソッドに置き換える！）
        for detection in detections:

            # BBox近傍のOptical Flowを抽出
            #flow_region = extract_flow_around_bbox_archive2(flow, detection, margin=10, padding=0)
            flow_region = extract_flow_around_bbox(
                flow=flow,
                target_detection=detection,
                detections = detections,
                base_margin=8,
                base_padding=0,
                individual_settings=True,
                scale_factor=0.025,
                score_factor=9.0
            )
            #print("flow_region shape:", flow_region.shape)


            # 抽出したOptical Flowの合成ベクトルを算出
            correction_vector = calculate_correction_vector_with_mean(flow_region)
            #correction_vector = calculate_correction_vector_with_histogram(flow_region, bins=15)
            #print("Correction vector (x, y):", correction_vector, "Shape:", correction_vector.shape) # Shape: (2,)でOK!

            # 可視化（近傍Optical Flow : hsv）


            # 可視化（近傍Optical Flow : 矢印）


            # 可視化（合成ベクトル）            


            # 位置情報を修正
            adjust_bbox_position(detection, correction_vector, weight=1.0, img_width=width, img_height=height)

        # ---------------------------------------------------------------------------------------------

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

# 自作メソッド---------------------------------------------------------------------------------------------------------------------

def calculate_margin_padding(detection, base_margin=5, base_padding=0, scale_factor=0.05, score_factor=3.0):
    """
        各物体のBBoxに応じたmarginとpaddingを計算する関数

        Parameters:
        - detection (STrack Object) : BBox情報を含むSTrackオブジェクト
        - base_margin (int) : 基準となるmargin値
        - base_padding (int) : 基準となるpadding値
        - scale_factor (float) : 面積や幅・高さに比例するスケール係数
        - score_factor (float) : 信頼度スコアに基づく補正係数（大きいほどスコアの影響を強くする）

        Returns:
        - margin (int) : 計算されたmargin値
        - padding (int) : 計算されたpadding値
    """
    # BBoxの幅と高さを取得
    _, _, w, h = detection.tlwh

    # 面積に基づいてmarginを設定
    area = w * h
    margin = base_margin + int(area * scale_factor)

    # スコアに基づいてpaddingを調整（信頼度が高いと小さくなる）
    score = getattr(detection, "score", 1.0)  # デフォルト値1.0（スコアが存在しない場合）
    score = max(0.0, min(score, 1.0))  # スコアは[0, 1]の範囲にクリップ
    padding = base_padding + int((1 - score) * score_factor)

    return margin, padding

def extract_flow_around_bbox(flow, target_detection, detections, base_margin=5, base_padding=0, individual_settings=False, scale_factor=0.1, score_factor=3.0):
    """
        BBox近傍のOptical Flowを抽出するメソッド

        Parameters:
        - flow (ndarray) : Optical Flow (形状: 高さ x 幅 x 2)
        - target_detection (STrack Object) : 補正対象物体のSTrackオブジェクト
        - detections (list of STrack) : 全ての物体のBBox情報リスト（STrackオブジェクトのリスト）
        - base_margin (int) : 全物体に一律で適用する基準margin値（個別設定しない場合）
        - base_padding (int) : 全物体に一律で適用する基準padding値（個別設定しない場合）
        - individual_settings (bool) : 個別にmarginとpaddingを設定するかのフラグ
        - scale_factor (float) : 個別設定時に面積や幅・高さに比例するスケール係数

        Returns:
        - flow_region (ndarray) : 近傍領域の背景部分から抽出したOptical Flow（形状: N x 2）
    """

    if individual_settings:
        margin, padding = calculate_margin_padding(target_detection, base_margin, base_padding, scale_factor, score_factor)
    else:
        margin, padding = base_margin, base_padding

    # 対象物体のBBox座標を取得（整数型に変換）
    x, y, w, h = map(int, target_detection.tlwh)

    # 画像の範囲内にクリップ（外側近傍）
    height, width, _ = flow.shape
    x1_outer = max(0, x - margin)
    y1_outer = max(0, y - margin)
    x2_outer = min(width, x + w + margin)
    y2_outer = min(height, y + h + margin)

    # 画像の範囲内にクリップ（対象物体の BBox + padding）
    x1_inner = max(0, x - padding)
    y1_inner = max(0, y - padding)
    x2_inner = min(width, x + w + padding)
    y2_inner = min(height, y + h + padding)

    # 初期マスクを作成（全て背景として初期化）
    mask = np.ones((y2_outer - y1_outer, x2_outer - x1_outer), dtype=bool)

    # 対象物体の BBox + padding を除外
    mask[
        (y1_inner - y1_outer):(y2_inner - y1_outer),
        (x1_inner - x1_outer):(x2_inner - x1_outer)
    ] = False

    # 他の物体の BBox + padding を除外
    for detection in detections:

        # 他の物体と対象物体を区別
        if detection is target_detection:
            continue

        # 他の物体の padding を個別に計算
        if individual_settings:
            _, other_padding = calculate_margin_padding(detection, base_margin, base_padding, scale_factor)
        else:
            other_padding = base_padding

        # 他の物体の BBox 座標を取得
        ox, oy, ow, oh = map(int, detection.tlwh)

        # 他の物体の BBox + padding 範囲
        ox1_inner = max(0, ox - other_padding)
        oy1_inner = max(0, oy - other_padding)
        ox2_inner = min(width, ox + ow + other_padding)
        oy2_inner = min(height, oy + oh + other_padding)

        # マスク内のインデックス範囲を計算して除外
        mask[
            max(0, oy1_inner - y1_outer):min(y2_outer - y1_outer, oy2_inner - y1_outer),
            max(0, ox1_inner - x1_outer):min(x2_outer - x1_outer, ox2_inner - x1_outer)
        ] = False

    # 近傍領域の背景部分から Optical Flow を抽出
    flow_region = flow[y1_outer:y2_outer, x1_outer:x2_outer][mask]

    return flow_region

def calculate_correction_vector_with_mean(flow_region):
    """
        Optical Flowの合成ベクトルを用いて補正量を算出するメソッド

        Parameters:
        - flow_region (ndarray) : BBox近傍から抽出したOptical Flow (N x 2)

        Returns:
        - composite_vector (ndarray) : 合成ベクトル (形状: (2,))
    """

    if flow_region.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    # 合成ベクトルを計算（平均）
    composite_vector = np.mean(flow_region, axis=0) # 1次元化されたデータに沿った平均

    return composite_vector


def calculate_correction_vector_with_histogram(flow_region, bins='auto'):
    """
        Optical Flowのヒストグラムを用いて補正量を直交座標で算出するメソッド

        Parameters:
        - flow_region (ndarray) : BBox近傍から抽出したOptical Flow (N x 2)
        - bins (int or str) : ヒストグラムのビン数 ('auto' も指定可能)

        Returns:
        - correction_vector (ndarray) : 補正量ベクトル (形状: (2,))
    """
    if flow_region.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    # x, y成分を分離
    x_values = flow_region[:, 0]
    y_values = flow_region[:, 1]

    # データ範囲の確認とクリップ
    x_values = np.clip(x_values, -1000, 1000)  # 必要に応じて範囲を調整
    y_values = np.clip(y_values, -1000, 1000)

    # データ型をfloat32に変換
    x_values = x_values.astype(np.float32)
    y_values = y_values.astype(np.float32)

    # x, y座標のヒストグラムを計算
    x_hist, x_bin_edges = np.histogram(x_values, bins=bins)
    y_hist, y_bin_edges = np.histogram(y_values, bins=bins)

    # 最頻値を取得（ヒストグラムの最大値のインデックスを用いる）
    x_mode_bin = np.argmax(x_hist)
    y_mode_bin = np.argmax(y_hist)

    # データ範囲の確認
    #print(f"Before clipping: x_values min={np.min(x_values)}, max={np.max(x_values)}, y_values min={np.min(y_values)}, max={np.max(y_values)}")

    # ビンの中央値を最頻値とする
    x_mode = (x_bin_edges[x_mode_bin] + x_bin_edges[x_mode_bin + 1]) / 2
    y_mode = (y_bin_edges[y_mode_bin] + y_bin_edges[y_mode_bin + 1]) / 2

    # 補正ベクトルを返す
    correction_vector = np.array([x_mode, y_mode], dtype=np.float32)

    return correction_vector

def calculate_correction_vector_with_polar(flow_region, bins='auto'):
    """
        Optical Flowのヒストグラムを用いて補正量を直交座標で算出するメソッド

        Parameters:
        - flow_region (ndarray) : BBox近傍から抽出したOptical Flow (N x 2)
        - bins (int or str) : ヒストグラムのビン数 ('auto' も指定可能)

        Returns:
        - correction_vector (ndarray) : 補正量ベクトル (形状: (2,))
    """

def adjust_bbox_position(detection, correction_vector, weight=1.0, img_width=None, img_height=None):
    """
        BBoxの位置情報を補正するメソッド

        Parameters:
        - detection (STrack Object) : 補正対象のSTrackオブジェクト
        - correction_vector (ndarray) : 補正量（2次元ベクトル）
        - weight (float) : 補正量を調整する重み
    """

    # 型チェック
    if not isinstance(correction_vector, np.ndarray) or correction_vector.shape != (2,):
        raise ValueError("correction_vector must be a NumPy array with shape (2,)")

    if weight < 0:
        raise ValueError("weight must be non-negative")

    # BBoxの位置情報（tlwh形式）を取得
    x, y, w, h = detection.tlwh

    # 補正量をアンパック
    dx , dy = correction_vector

    # x, yを補正
    x -= weight * dx
    y -= weight * dy

    # 範囲外をクリップ
    if img_width is not None:
        x = max(0, min(x, img_width - w))
    if img_height is not None:
        y = max(0, min(y, img_height - h))

    # 更新されたtlwhを反映
    detection._tlwh = np.array([x, y, w, h], dtype=np.float32)

def visualize_flow_hsv(img_file, flow):
    """
        算出したOptical FlowをHSV色空間で可視化するメソッド
    """

    frame = cv2.imread(img_file)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255  # 彩度を最大（255）に設定
    frame_height, frame_width = cv2.imread(img_file).shape[:2]

    # 流速（magnitude）と角度（angle）を計算
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # HSV画像を作成
    hsv[..., 0] = ang * 180 / np.pi / 2  # 色相（範囲は[0, 179]）
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 明度（範囲は[0, 255]）
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # HSV -> BGR変換

def visualize_flow_arrow():
    """
        算出したOptical FlowをHSV色空間で可視化するメソッド
    """
    print()

def visualize_composite_vector():
    """
        算出した合成ベクトルを矢印で可視化するメソッド
    """
    print()

# 自作メソッド（アーカイブ）-------------------------------------------------------------------------------------------------

def extract_flow_around_bbox_archive1(flow, detection, margin=5):
    """
        BBox近傍のOptical Flowを抽出するメソッド (BBox領域内もすべて抽出する)

        Parameters:
        - flow (ndarray) : Optical Flow
        - detection (STrack Object) : 補正対象のSTrackオブジェクト
        - margin (int) : Optical Flowを抽出するBBoxの近傍の大きさ

        Returns:
        - flow_region (ndarray) : BBox近傍から抽出したOptical Flow
    """

    # BBox座標を取得（整数型に変換）
    x, y, w, h = map(int, detection.tlwh)

    # 画像の範囲内にクリップ
    height, width, _ = flow.shape
    x1 = max(0, x-margin)
    y1 = max(0, y-margin)
    x2 = min(width, x+width+margin)
    y2 = min(height, y+height+margin)

    # BBox近傍のOptical Flowを抽出
    flow_region = flow[y1:y2, x1:x2]

    return flow_region

def extract_flow_around_bbox_archive2(flow, detection, margin=5, padding=0):
    """
        BBox近傍のOptical Flowを抽出するメソッド（BBox + padding領域内は抽出しない）

        Parameters:
        - flow (ndarray) : Optical Flow (形状: 高さ x 幅 x 2)
        - detection (STrack Object) : 補正対象のSTrackオブジェクト
        - margin (int) : Optical Flowを抽出するBBoxの近傍の大きさ
        - padding (int): BBoxと外側近傍の間隔（背景部分のみ抽出するための余白）

        Returns:
        - flow_region (ndarray) : BBox外側の近傍から抽出したOptical Flow（形状: N x 2）
    """

    # BBox座標を取得（整数型に変換）
    x, y, w, h = map(int, detection.tlwh)

    # 画像の範囲内にクリップ（外側近傍）
    height, width, _ = flow.shape
    x1_outer = max(0, x - margin)
    y1_outer = max(0, y - margin)
    x2_outer = min(width, x + w + margin)
    y2_outer = min(height, y + h + margin)

    # 画像の範囲内にクリップ（BBox + padding）
    x1_inner = max(0, x - padding)
    y1_inner = max(0, y - padding)
    x2_inner = min(width, x + w + padding)
    y2_inner = min(height, y + h + padding)

    # Optical Flowのマスクを作成（BBoxとその周囲paddingを除外）
    mask = np.ones((y2_outer - y1_outer, x2_outer - x1_outer), dtype=bool)
    mask[
        (y1_inner - y1_outer):(y2_inner - y1_outer),
        (x1_inner - x1_outer):(x2_inner - x1_outer)
    ] = False

    # BBox外側の背景部分Optical Flowを抽出
    flow_region = flow[y1_outer:y2_outer, x1_outer:x2_outer][mask]

    return flow_region

def calculate_correction_vector_with_mean_archive1(flow_region):
    """
        与えられたベクトル群の合成ベクトルを求めるメソッド

        Parameters:
        - flow_region (ndarray) : BBox近傍から抽出したOptical Flow (高さ x 幅 x 2)

        Returns:
        - correction_vector (ndarray) : 合成ベクトル (形状: (2,))
    """

    if flow_region.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    # 合成ベクトルを計算（平均）
    correction_vector = np.mean(flow_region, axis=(0, 1))

    return correction_vector

