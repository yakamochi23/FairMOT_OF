from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

import slackweb

from lib.tracker.multitracker import JDETracker
from lib.tracker.multitracker_with_of import JDETracker_OF
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
import lib.datasets.dataset.jde as datasets

from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts

from datetime import datetime


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)

    #tracker = JDETracker(opt, frame_rate=frame_rate) # ベースモデル（FairMOT）
    tracker = JDETracker_OF(opt, frame_rate=frame_rate) # 提案手法
    timer = Timer()
    results = []
    frame_id = 0

    # 追加-------------------------------------------------------------------------------------------

    #save_optical_dir = 

    # 最初のフレームimg0だけ取得（path と img は無視）
    _, _, img0 = next(iter(dataloader))
    prev_img0 = img0
    # -----------------------------------------------------------------------------------------------

    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        # 変更----------------------------------------------------------------------------------------

        # 既存手法
        #online_targets = tracker.update(blob, img0)


        # 提案手法
        flow = calculate_optical_flow(prev_img0, img0) # 隣接2フレームでOptical Flowを算出
        online_targets = tracker.update(blob, img0, flow)

        # --------------------------------------------------------------------------------------------

        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        frame_id += 1
        prev_img0 = img0

    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls

def calculate_optical_flow(old_frame, frame):
    """
        隣接2フレームでOptical Flowを計算するメソッド

        Parameters:
        - old_frame (ndarray) : 前のフレーム
        - frame (ndarray) : 現在のフレーム

        Returns:
        - flow (ndarray) : 算出したOptical Flow
    """

    # フレームの整合性チェック
    if old_frame.shape[:2] != frame.shape[:2]:
        raise ValueError("The two input frames must have the same dimensions.")
    if old_frame.dtype != np.uint8 or frame.dtype != np.uint8:
        raise ValueError("Input frames must have dtype uint8.")

    # グレースケールに変換
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) if len(old_frame.shape) == 3 else old_frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Dense Optical Flowを計算
    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0)

    if flow is None:
        raise RuntimeError("Optical flow calculation failed.")

    return flow

def main(opt, data_root='/data/MOT16/train', dst_root='E:/',det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    #result_root = os.path.join(data_root, '..', 'results', exp_name)
    result_root = os.path.join(dst_root, exp_name, 'output_result')
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        #output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        output_dir = os.path.join(dst_root, exp_name, 'output_viz', seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))

        # MOTchallenge ------------------------------------------------------------------------------------
        #meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        #frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        # -------------------------------------------------------------------------------------------------

        # VisDrone or UAVDT -------------------------------------------------------------------------------
        frame_rate = 30
        # -------------------------------------------------------------------------------------------------

        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            #cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
        dst_root = 'E:/Results/MOT15/val'
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/train')
        dst_root = 'E:/Results/MOT16/val'
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/test')
        dst_root = 'E:/Results/MOT16/test'
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
        dst_root = 'E:/Results/MOT15/test'
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
        dst_root = 'E:/Results/MOT17/test'
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
        dst_root = 'E:/Results/MOT17/val'
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
        dst_root = 'E:/Results/MOT15/val'
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
        dst_root = 'E:/Results/MOT20/val'
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
        dst_root = 'E:/Results/MOT20/test'
    if opt.val_visdrone_pedi:
        seqs_str = '''uav0000086_00000_v
                      uav0000117_02622_v
                      uav0000137_00458_v
                      uav0000182_00000_v
                      uav0000268_05773_v
                      uav0000305_00000_v
                      uav0000339_00001_v
                      '''
        data_root = os.path.join(opt.data_dir, 'VisDrone_pedi/images/val')
        dst_root = 'E:/Results/VisDrone_pedi/val'
    if  opt.test_visdrone_pedi:
        seqs_str = '''uav0000009_03358_v
                      uav0000073_00600_v
                      uav0000073_04464_v
                      uav0000077_00720_v
                      uav0000088_00290_v
                      uav0000119_02301_v
                      uav0000120_04775_v
                      uav0000161_00000_v
                      uav0000188_00000_v
                      uav0000201_00000_v
                      uav0000249_00001_v
                      uav0000249_02688_v
                      uav0000297_00000_v
                      uav0000297_02761_v
                      uav0000306_00230_v
                      uav0000355_00001_v
                      uav0000370_00001_v
                      '''
        data_root = os.path.join(opt.data_dir, 'VisDrone_pedi/images/test')
        dst_root = 'E:/Results/VisDrone_pedi/test'
    if opt.val_visdrone_car:
        seqs_str = '''uav0000086_00000_v
                      uav0000117_02622_v
                      uav0000137_00458_v
                      uav0000182_00000_v
                      uav0000268_05773_v
                      uav0000305_00000_v
                      uav0000339_00001_v
                      '''
        data_root = os.path.join(opt.data_dir, 'VisDrone_car/images/val')
        dst_root = 'E:/Results/VisDrone_car/val'
    if  opt.test_visdrone_car:
        seqs_str = '''uav0000009_03358_v
                      uav0000073_00600_v
                      uav0000073_04464_v
                      uav0000077_00720_v
                      uav0000088_00290_v
                      uav0000119_02301_v
                      uav0000120_04775_v
                      uav0000161_00000_v
                      uav0000188_00000_v
                      uav0000201_00000_v
                      uav0000249_00001_v
                      uav0000249_02688_v
                      uav0000297_00000_v
                      uav0000297_02761_v
                      uav0000306_00230_v
                      uav0000355_00001_v
                      uav0000370_00001_v
                      '''
        data_root = os.path.join(opt.data_dir, 'VisDrone_car/images/test')
        dst_root = 'E:/Results/VisDrone_car/test'
    if  opt.train_uavdt_car:
        seqs_str = '''M0101
                      M0201
                      M0202
                      M0204
                      M0206
                      M0207
                      M0210
                      M0301
                      M0401
                      M0402
                      M0501
                      M0603
                      M0604
                      M0605
                      M0702
                      M0703
                      M0704
                      M0901
                      M0902
                      M1002
                      M1003
                      M1005
                      M1006
                      M1008
                      M1102
                      M1201
                      M1202
                      M1304
                      M1305
                      M1306
                      '''
        data_root = os.path.join(opt.data_dir, 'UAVDT_car/images/train')
        dst_root = 'E:/Results/UAVDT_car/train'
    if  opt.test_uavdt_car:
        seqs_str = '''M0203
                      M0205
                      M0208
                      M0209
                      M0403
                      M0601
                      M0602
                      M0606
                      M0701
                      M0801
                      M0802
                      M1001
                      M1004
                      M1007
                      M1009
                      M1101
                      M1301
                      M1302
                      M1303
                      M1401
                      '''
        data_root = os.path.join(opt.data_dir, 'UAVDT_car/images/test')
        dst_root = 'E:/Results/UAVDT_car/test'
    if opt.custom:
        data_root = os.path.join(opt.data_dir, 'Custom/images')
        dst_root = 'E:/Results/Custom'

    seqs = [seq.strip() for seq in seqs_str.split()] # 通常
    #seqs = [s for s in os.listdir(data_root)] # Customデータセットを使用する場合

    print(seqs)

    # 実験名に現在時刻を設定
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # 現在時刻を取得
    exp_name = f'exp_{current_time}'

    # 通知を飛ばすslackチャンネルの設定
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")

    # 通知メッセージ
    slack.notify(text="Program has Started!")

    main(opt,
         data_root=data_root,
         dst_root=dst_root,
         seqs=seqs,
         exp_name=exp_name,
         show_image=False,
         save_images=True,
         save_videos=True)
    
    # 通知メッセージ
    slack.notify(text="Program has ended!")
