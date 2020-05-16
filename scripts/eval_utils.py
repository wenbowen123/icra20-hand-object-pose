import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import argparse
import open3d as O
from PIL import Image, ImageDraw
from scipy import spatial
from scipy.spatial import distance

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir+'/../')

#https://github.com/thodan/sixd_toolkit
def ensure_dir(path):
    """
    Ensures that the specified directory exists.

    :param path: Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def draw_rect(vis, rect, color=(255, 255, 255)):
    vis_pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(vis_pil)
    draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                   outline=color, fill=None)
    del draw
    return np.asarray(vis_pil)

def project_pts(pts, K, R, t):
    assert(pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T

def norm_depth(depth, valid_start=0.2, valid_end=1.0):
    mask = depth > 0
    depth_n = depth.astype(np.float)
    depth_n[mask] -= depth_n[mask].min()
    depth_n[mask] /= depth_n[mask].max() / (valid_end - valid_start)
    depth_n[mask] += valid_start
    return depth_n

def depth_im_to_dist_im(depth_im, K):
    """
    Converts depth image to distance image.

    :param depth_im: Input depth image, where depth_im[y, x] is the Z coordinate
    of the 3D point [X, Y, Z] that projects to pixel [x, y], or 0 if there is
    no such 3D point (this is a typical output of the Kinect-like sensors).
    :param K: Camera matrix.
    :return: Distance image dist_im, where dist_im[y, x] is the distance from
    the camera center to the 3D point [X, Y, Z] that projects to pixel [x, y],
    or 0 if there is no such 3D point.
    """
    xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
    ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

    dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)
    return dist_im

def rgbd_to_point_cloud(K, depth, rgb=np.array([])):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    pts_im = np.vstack([us, vs]).T
    if rgb != np.array([]):
        colors = rgb[vs, us, :]
    else:
        colors = None
    return pts, colors, pts_im

def clip_pt_to_im(pt, im_size):
    pt_c = [min(max(pt[0], 0), im_size[0] - 1),
            min(max(pt[1], 0), im_size[1] - 1)]
    return pt_c

def calc_2d_bbox(xs, ys, im_size=None, clip=False):
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    if clip:
        assert(im_size is not None)
        bb_tl = clip_pt_to_im(bb_tl, im_size)
        bb_br = clip_pt_to_im(bb_br, im_size)
    return [bb_tl[0], bb_tl[1], bb_br[0] - bb_tl[0], bb_br[1] - bb_tl[1]]

def calc_pose_2d_bbox(model, im_size, K, R_m2c, t_m2c):
    pts_im = project_pts(model['pts'], K, R_m2c, t_m2c)
    pts_im = np.round(pts_im).astype(np.int)
    return calc_2d_bbox(pts_im[:, 0], pts_im[:, 1], im_size)

def crop_im(im, roi):
    if im.ndim == 3:
        crop = im[max(roi[1], 0):min(roi[1] + roi[3] + 1, im.shape[0]),
               max(roi[0], 0):min(roi[0] + roi[2] + 1, im.shape[1]), :]
    else:
        crop = im[max(roi[1], 0):min(roi[1] + roi[3] + 1, im.shape[0]),
               max(roi[0], 0):min(roi[0] + roi[2] + 1, im.shape[1])]
    return crop

def paste_im(src, trg, pos):
    """
    Pastes src to trg with the top left corner at pos.
    """
    assert(src.ndim == trg.ndim)

    # Size of the region to be pasted
    w = min(src.shape[1], trg.shape[1] - pos[0])
    h = min(src.shape[0], trg.shape[0] - pos[1])

    if src.ndim == 3:
        trg[pos[1]:(pos[1] + h), pos[0]:(pos[0] + w), :] = src[:h, :w, :]
    else:
        trg[pos[1]:(pos[1] + h), pos[0]:(pos[0] + w)] = src[:h, :w]

def paste_im_mask(src, trg, pos, mask):
    assert(src.ndim == trg.ndim)
    assert(src.shape[:2] == mask.shape[:2])
    src_pil = Image.fromarray(src)
    trg_pil = Image.fromarray(trg)
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    trg_pil.paste(src_pil, pos, mask_pil)
    trg[:] = np.array(trg_pil)[:]

def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T

def calc_pts_diameter(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set).

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    diameter = -1
    for pt_id in range(pts.shape[0]):
        #if pt_id % 1000 == 0: print(pt_id)
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def calc_pts_diameter2(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set). Faster but requires more memory than
    calc_pts_diameter.

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    dists = distance.cdist(pts, pts, 'euclidean')
    diameter = np.max(dists)
    return diameter


def adi(R_est, t_est, R_gt, t_gt, model):
  """
  Average Distance of Model Points for objects with indistinguishable views
  - by Hinterstoisser et al. (ACCV 2012).

  :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
  :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
  :param model: Object model given by a dictionary where item 'pts'
  is nx3 ndarray with 3D model points.
  :return: Error of pose_est w.r.t. pose_gt.
  """
  pts_est = transform_pts_Rt(model['pts'], R_est, t_est)
  pts_gt = transform_pts_Rt(model['pts'], R_gt, t_gt)

  # Calculate distances to the nearest neighbors from pts_gt to pts_est
  nn_index = spatial.cKDTree(pts_est)
  nn_dists, _ = nn_index.query(pts_gt, k=1)

  e = nn_dists.mean()
  return e

def re(R_est, R_gt):
  """
  Rotational Error.

  :param R_est: Rotational element of the estimated pose (3x1 vector).
  :param R_gt: Rotational element of the ground truth pose (3x1 vector).
  :return: Error of t_est w.r.t. t_gt.
  """
  assert(R_est.shape == R_gt.shape == (3, 3))
  error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
  error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
  error = math.acos(error_cos)
  error = 180.0 * error / np.pi # [rad] -> [deg]
  return error

def te(t_est, t_gt):
  """
  Translational Error.

  :param t_est: Translation element of the estimated pose (3x1 vector).
  :param t_gt: Translation element of the ground truth pose (3x1 vector).
  :return: Error of t_est w.r.t. t_gt.
  """
  assert(t_est.size == t_gt.size == 3)
  error = np.linalg.norm(t_gt - t_est)
  return error



def load_ply(path):
  """
  Loads a 3D mesh model from a PLY file.

  :param path: Path to a PLY file.
  :return: The loaded model given by a dictionary with items:
  'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
  'faces' (mx3 ndarray) - the latter three are optional.
  """
  f = open(path, 'r')

  n_pts = 0
  n_faces = 0
  face_n_corners = 3 # Only triangular faces are supported
  pt_props = []
  face_props = []
  is_binary = False
  header_vertex_section = False
  header_face_section = False

  # Read header
  while True:
    line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
    if line.startswith('element vertex'):
      n_pts = int(line.split()[-1])
      header_vertex_section = True
      header_face_section = False
    elif line.startswith('element face'):
      n_faces = int(line.split()[-1])
      header_vertex_section = False
      header_face_section = True
    elif line.startswith('element'): # Some other element
      header_vertex_section = False
      header_face_section = False
    elif line.startswith('property') and header_vertex_section:
      # (name of the property, data type)
      pt_props.append((line.split()[-1], line.split()[-2]))
    elif line.startswith('property list') and header_face_section:
      elems = line.split()
      if elems[-1] == 'vertex_indices':
        # (name of the property, data type)
        face_props.append(('n_corners', elems[2]))
        for i in range(face_n_corners):
          face_props.append(('ind_' + str(i), elems[3]))
      else:
        print('Warning: Not supported face property: ' + elems[-1])
    elif line.startswith('format'):
      if 'binary' in line:
        is_binary = True
    elif line.startswith('end_header'):
      break

  # Prepare data structures
  model = {}
  model['pts'] = np.zeros((n_pts, 3), np.float)
  if n_faces > 0:
    model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

  pt_props_names = [p[0] for p in pt_props]
  is_normal = False
  if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
    is_normal = True
    model['normals'] = np.zeros((n_pts, 3), np.float)

  is_color = False
  if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
    is_color = True
    model['colors'] = np.zeros((n_pts, 3), np.float)

  is_texture = False
  if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
    is_texture = True
    model['texture_uv'] = np.zeros((n_pts, 2), np.float)

  formats = { # For binary format
    'float': ('f', 4),
    'double': ('d', 8),
    'int': ('i', 4),
    'uchar': ('B', 1)
  }

  # Load vertices
  for pt_id in range(n_pts):
    prop_vals = {}
    load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
            'red', 'green', 'blue', 'texture_u', 'texture_v']
    if is_binary:
      for prop in pt_props:
        format = formats[prop[1]]
        val = struct.unpack(format[0], f.read(format[1]))[0]
        if prop[0] in load_props:
          prop_vals[prop[0]] = val
    else:
      elems = f.readline().rstrip('\n').rstrip('\r').split()
      for prop_id, prop in enumerate(pt_props):
        if prop[0] in load_props:
          prop_vals[prop[0]] = elems[prop_id]

    model['pts'][pt_id, 0] = float(prop_vals['x'])
    model['pts'][pt_id, 1] = float(prop_vals['y'])
    model['pts'][pt_id, 2] = float(prop_vals['z'])

    if is_normal:
      model['normals'][pt_id, 0] = float(prop_vals['nx'])
      model['normals'][pt_id, 1] = float(prop_vals['ny'])
      model['normals'][pt_id, 2] = float(prop_vals['nz'])

    if is_color:
      model['colors'][pt_id, 0] = float(prop_vals['red'])
      model['colors'][pt_id, 1] = float(prop_vals['green'])
      model['colors'][pt_id, 2] = float(prop_vals['blue'])

    if is_texture:
      model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
      model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

  # Load faces
  for face_id in range(n_faces):
    prop_vals = {}
    if is_binary:
      for prop in face_props:
        format = formats[prop[1]]
        val = struct.unpack(format[0], f.read(format[1]))[0]
        if prop[0] == 'n_corners':
          if val != face_n_corners:
            print('Error: Only triangular faces are supported.')
            print('Number of face corners: ' + str(val))
            exit(-1)
        else:
          prop_vals[prop[0]] = val
    else:
      elems = f.readline().rstrip('\n').rstrip('\r').split()
      for prop_id, prop in enumerate(face_props):
        if prop[0] == 'n_corners':
          if int(elems[prop_id]) != face_n_corners:
            print('Error: Only triangular faces are supported.')
            print('Number of face corners: ' + str(int(elems[prop_id])))
            exit(-1)
        else:
          prop_vals[prop[0]] = elems[prop_id]

    model['faces'][face_id, 0] = int(prop_vals['ind_0'])
    model['faces'][face_id, 1] = int(prop_vals['ind_1'])
    model['faces'][face_id, 2] = int(prop_vals['ind_2'])

  f.close()

  return model


