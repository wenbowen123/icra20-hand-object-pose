import os,sys,re,copy
import numpy as np
import glob
import eval_utils as EU
import open3d as O
import argparse
import collections
import pickle
from collections import defaultdict

def evalAll():
  rgb_files = []
  for dirName, subdirList, fileList in os.walk('/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/catkin_ws/src/icra20_manipulation_pose/motoman_data/auto_collect/'):
    for fname in fileList:
      if 'rgb' in fname:
        rgb_files.append(os.path.join(dirName,fname))
  rgb_files.sort()

  objects = ['cylinder','cuboid','ellipse','tless3']
  models = {}
  pcds = {}
  for ob in objects:
    pcd = O.io.read_point_cloud("/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/catkin_ws/src/icra20_manipulation_pose/meshes/{}.ply".format(ob))
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    pts = np.asarray(pcd.points).copy()
    pcds[ob] = pcd
    model={}
    model['pts'] = pts
    models[ob] = model
    print('model {} pts: {}'.format(ob, pts.shape))

  errs = {}
  for ob in objects:
    errs[ob] = []

  for i in range(len(rgb_files)):
    rgb_file = rgb_files[i]
    print('rgb_file:',rgb_file)
    dir = os.path.dirname(rgb_file)+'/'
    index = os.path.basename(rgb_file).split('.')[0].replace('rgb','')
    gt_file = dir+'refined_gt/ob_in_cam{}.txt'.format(index)

    cur_model = None
    cur_ob = None
    for ob in objects:
      if ob in gt_file:
        cur_ob = copy.deepcopy(ob)
        cur_model = models[ob]
        break
    if cur_model is None:
      print('ERROR no corresponding model found')
      continue

    pred_file = os.path.dirname(gt_file)+'/../predict/{}/model2scene.txt'.format(index)
    if os.path.exists(pred_file):
      pred = np.loadtxt(pred_file)
    else:
      print('ERROR {} not found'.format(pred_file))
      pred = np.eye(4)

    if os.path.exists(gt_file):
      gt = np.loadtxt(gt_file)
    else:
      print('ERROR {} not found'.format(gt_file))
      gt = np.eye(4)

    R_e = pred[:3,:3]
    t_e = pred[:3,3]
    R_g = gt[:3,:3]
    t_g = gt[:3,3]
    e = EU.adi(R_e, t_e, R_g, t_g, cur_model)
    errs[cur_ob].append(e)
    print('object={}, err={}'.format(cur_ob,e))

  for ob in errs.keys():
    errs[ob] = np.array(errs[ob])
    recall = np.sum(errs[ob]<0.005)/len(errs[ob])
    print('object={}, total={}, recall={}'.format(ob,len(errs[ob]),recall))



if __name__=='__main__':
  evalAll()





