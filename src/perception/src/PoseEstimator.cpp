#include "PoseEstimator.h"
#include <pcl/registration/super4pcs.h>
#include <gr/shared.h>



template<class PointT>
PoseEstimator<PointT>::PoseEstimator(ConfigParser *cfg1, boost::shared_ptr<pcl::PointCloud<PointT> > model, boost::shared_ptr<pcl::PointCloud<PointT> > model001, const Eigen::Matrix3f &cam_K): _cam_K(cam_K),cfg(cfg1), _renderer(480,640, cam_K(0,0), cam_K(1,1), cam_K(0,2), cam_K(1,2))
{
  _model = model;
  _model001 = model001;
  pcl::computeCentroid<PointT, pcl::PointXYZRGBNormal>(*_model001, _model_center_init);
  Eigen::Vector4f max_pt(0,0,0,1), min_pt(0,0,0,1);
  pcl::getMinMax3D<PointT>(*_model001, min_pt, max_pt);
  _model_radius = (max_pt-min_pt).norm()/2.0;
  PointT minPt, maxPt;
  pcl::getMinMax3D (*_model001, minPt, maxPt);
  _smallest_dim = std::min(std::min((maxPt.x-minPt.x),(maxPt.y-minPt.y)), (maxPt.z-minPt.z));

  _ob_diameter = std::sqrt( (maxPt.x-minPt.x)*(maxPt.x-minPt.x) + ((maxPt.y-minPt.y))*((maxPt.y-minPt.y)) + ((maxPt.z-minPt.z))*((maxPt.z-minPt.z)) );
  _obj_mesh.reset(new pcl::PolygonMesh);
  pcl::io::loadOBJFile(cfg->object_mesh_path,*_obj_mesh);
}

template<class PointT>
PoseEstimator<PointT>::~PoseEstimator()
{

}

template<class PointT>
void PoseEstimator<PointT>::setCurScene(boost::shared_ptr<pcl::PointCloud<PointT> > scene, PointCloudRGBNormal::Ptr cloud_withouthand_raw, boost::shared_ptr<pcl::PointCloud<PointT> > object_segment, const cv::Mat &rgb, const cv::Mat &depth_meters)
{
  _depth_meters = depth_meters;
  _rgb = rgb;
  _object_segment = object_segment;
  _scene = scene;
  _scene_high_confidence = boost::make_shared<pcl::PointCloud<PointT> >();
  _cloud_withouthand_raw = cloud_withouthand_raw;

  for (const auto &pt:_object_segment->points)
  {
    if (pt.confidence<cfg->pose_estimator_high_confidence_thres) continue;  //Confidence about being object
    _scene_high_confidence->points.push_back(pt);
  }
}

template<class PointT>
void PoseEstimator<PointT>::reset()
{
  _scene->clear();
  _scene_high_confidence->clear();
  _object_segment->clear();
  _cloud_withouthand_raw->clear();
  _depth_meters.release();
  _rgb.release();
  _sdf.reset();
  _pose_hypos.clear();
  _renderer.reset();
}

template<class PointT>
bool PoseEstimator<PointT>::runSuper4pcs(const std::map<std::vector<int>, std::vector<std::pair<int, int> > > &ppfs)
{
  pcl::Super4PCS<PointT,PointT> super4pcs;
  super4pcs.options_.sample_size = cfg->super4pcs_sample_size;
  super4pcs.options_.configureOverlap(cfg->super4pcs_overlap);
  super4pcs.options_.max_time_seconds = cfg->super4pcs_max_time_seconds;
  super4pcs.options_.delta = cfg->super4pcs_delta;
  super4pcs.options_.sample_dispersion = cfg->yml["super4pcs_dispersion"].as<float>();
  super4pcs.options_.success_quadrilaterals = cfg->yml["super4pcs_success_quadrilaterals"].as<int>();
  super4pcs.options_.max_normal_difference = cfg->super4pcs_max_normal_difference;
  super4pcs.options_.max_color_distance = cfg->super4pcs_max_color_distance;

  super4pcs.setPPFHash(ppfs);
  super4pcs.setInputSource (_model);    // Q
  super4pcs.setInputTarget (_scene_high_confidence);    // P, we extracted base on scene
  boost::shared_ptr<pcl::PointCloud<PointT> > aligned(new pcl::PointCloud<PointT>);
  super4pcs.align (*aligned);

  std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > hypos;
  std::vector<float> scores;
  super4pcs.getPoseHypo(hypos,scores);
  assert(hypos.size()==scores.size());

  if (hypos.size()==0)
  {
    return false;
  }

  _pose_hypos.clear();
  _pose_hypos.reserve(hypos.size());


  for (int i=0;i<hypos.size();i++)
  {
    _pose_hypos.push_back(PoseHypo(hypos[i], i, scores[i]));
  }
  return true;
}



//@angle_diff: unit is degree
//@dist_diff: unit is meter
template<class PointT>
void PoseEstimator<PointT>::clusterPoses(float angle_diff, float dist_diff, bool assign_id)
{
  printf("num original candidates = %d\n",_pose_hypos.size());
  class HypoCompare
  {
  public:
    bool operator () (const PoseHypo &p1, const PoseHypo &p2) // better one returns true
    {
      if (p1._lcp_score>p2._lcp_score) return true;
      if (p1._lcp_score<p2._lcp_score) return false;

      if (p1._id<p2._id) return true;
      if (p1._id>p2._id) return false;

      return false;
    };
  };

  //TODO: Will it really be faster to sort ???
  std::sort(_pose_hypos.begin(),_pose_hypos.end(), HypoCompare());


  std::vector<PoseHypo,Eigen::aligned_allocator<PoseHypo> > hypo_tmp = _pose_hypos;
  _pose_hypos.clear();
  _pose_hypos.push_back(hypo_tmp[0]);  // now it becomes the pose clusters

  const float radian_thres = angle_diff/180.0*M_PI;
  const std::string model_name = cfg->yml["model_name"].as<std::string>();
  const float x_symmetry = cfg->yml["object_symmetry"][model_name]["x"].as<double>()/180 * M_PI;
  const float y_symmetry = cfg->yml["object_symmetry"][model_name]["y"].as<double>()/180 * M_PI;
  const float z_symmetry = cfg->yml["object_symmetry"][model_name]["z"].as<double>()/180 * M_PI;

  printf("object symmetry: x=%f, y=%f, z=%f\n",x_symmetry, y_symmetry, z_symmetry);

  for (int i=1;i<hypo_tmp.size();i++)
  {
    bool isnew = true;
    Eigen::Matrix4f cur_pose=hypo_tmp[i]._pose;
    for (auto cluster:_pose_hypos)
    {
      Eigen::Vector3f t0 = cluster._pose.block(0,3,3,1);
      Eigen::Vector3f t1 = cur_pose.block(0,3,3,1);

      if ((t0-t1).norm()>=dist_diff)
      {
        continue;
      }

      Eigen::Matrix3f R0 = cluster._pose.block(0,0,3,3);   // rotate rpy around static axis
      Eigen::Vector3f rpy = R0.eulerAngles(2,1,0);
      float r0 = rpy(2);
      float p0 = rpy(1);
      float y0 = rpy(0);


      Eigen::Matrix3f R1 = cur_pose.block(0,0,3,3);
      Eigen::Vector3f rpy1 = R1.eulerAngles(2,1,0);
      float r1 = rpy1(2);
      float p1 = rpy1(1);
      float y1 = rpy1(0);



      float roll_diff = std::abs(r0-r1);
      float pitch_diff = std::abs(p0-p1);
      float yaw_diff = std::abs(y0-y1);

      if (x_symmetry==0)
      {
        roll_diff=0;
      }
     else if (x_symmetry>0)
      {
        roll_diff = std::min(roll_diff, static_cast<float>(x_symmetry) - roll_diff);
      }

      if (y_symmetry==0)
      {
        pitch_diff = 0;
      }
      else if (y_symmetry>0)
      {
        pitch_diff = std::min(pitch_diff, static_cast<float>(y_symmetry) - pitch_diff);
      }

      if (z_symmetry==0)
      {
        yaw_diff=0;
      }
      else if (z_symmetry>0)
      {
        yaw_diff = std::min(yaw_diff, static_cast<float>(z_symmetry) - yaw_diff);
      }

      if (pitch_diff<=radian_thres && roll_diff<=radian_thres && yaw_diff<=radian_thres)
      {
        isnew = false;
        break;
      }


      float rot_diff = Utils::rotationGeodesicDistance(R0, R1);
      if (rot_diff <= radian_thres)
      {
        isnew = false;
        break;
      }
    }

    if (isnew)
    {
      _pose_hypos.push_back(hypo_tmp[i]);
    }
  }

  if (assign_id)
  {
    for (int i=0;i<_pose_hypos.size();i++)
    {
      _pose_hypos[i]._id=i;
    }

  }

  printf("num of pose clusters: %d\n",_pose_hypos.size());

}

template<class PointT>
void PoseEstimator<PointT>::refineByICP()
{
  std::vector<PoseHypo,Eigen::aligned_allocator<PoseHypo> > hypo_tmp = _pose_hypos;
  _pose_hypos.clear();
  _pose_hypos.reserve(100);
  int num_candidates_keep = std::min(static_cast<int>(hypo_tmp.size()), 100);
  PointCloudRGBNormal::Ptr cloud_withouthand_raw_tmp(new PointCloudRGBNormal);
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(_cloud_withouthand_raw, cloud_withouthand_raw_tmp,0.005);
  const float icp_dist_thres = cfg->yml["icp_dist_thres"].as<float>();
  const float icp_angle_thres = cfg->yml["icp_angle_thres"].as<float>();

#pragma omp parallel
  {
    boost::shared_ptr<pcl::PointCloud<PointT> > scene_high_confidence(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*_scene_high_confidence, *scene_high_confidence);
    boost::shared_ptr<pcl::PointCloud<PointT> > model(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*_model, *model);
    boost::shared_ptr<pcl::PointCloud<PointT> > cloud_withouthand_raw(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud_withouthand_raw_tmp, *cloud_withouthand_raw);
    std::vector<float> scene_weights(cloud_withouthand_raw->points.size(), 1.0);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < num_candidates_keep; i++)
    {
      Eigen::Matrix4f model2scene(Eigen::Matrix4f::Identity());
      Eigen::Matrix4f transformation = hypo_tmp[i]._pose; // model2scene
      boost::shared_ptr<pcl::PointCloud<PointT>> model_4pcs(new pcl::PointCloud<PointT>);
      pcl::transformPointCloudWithNormals(*model, *model_4pcs, transformation);
      model2scene = transformation;

      float icp_score = Utils::runICP<pcl::PointSurfel>(scene_high_confidence, model_4pcs, transformation, 10, icp_angle_thres, icp_dist_thres); // scene2model
      model2scene = transformation.inverse() * model2scene;

      hypo_tmp[i]._pose = model2scene;

      #pragma omp critical
      _pose_hypos.push_back(hypo_tmp[i]);
    }
  }
}

template<class PointT>
void PoseEstimator<PointT>::projectCloudAndCompare(boost::shared_ptr<pcl::PointCloud<PointT> > cloud, cv::Mat &projection, float &avg_wrong_score, int id)
{

  const int H = 480, W = 640;
  projection = cv::Mat::zeros(480, 640, CV_32FC1);
  for (auto pt:cloud->points)
  {
    int u = _cam_K(0,0)*pt.x/pt.z + _cam_K(0,2);
    int v = _cam_K(1,1)*pt.y/pt.z + _cam_K(1,2);

    if (u<0 || u>=W || v<0 || v>=H)
    {
      continue;
    }
    if (projection.at<float>(v,u)==0)
    {
      projection.at<float>(v,u) = pt.z;
    }
    else
    {
      if (pt.z < projection.at<float>(v,u))
      {
        projection.at<float>(v,u) = pt.z;
      }
    }

  }

  //Compare with depth image
  float wrong_score = 0;
  cv::Mat projection_img = _rgb.clone();
  for (auto pt:cloud->points)
  {
    int u = _cam_K(0,0)*pt.x/pt.z + _cam_K(0,2);
    int v = _cam_K(1,1)*pt.y/pt.z + _cam_K(1,2);

    if (u<0 || u>=W || v<0 || v>=H)
    {
      continue;
    }
    if (_depth_meters.at<float>(v,u)<=0.1 || _depth_meters.at<float>(v,u)>=2.0)
    {
      wrong_score+=1;
      continue;
    }
    // not occluded
    float diff=std::abs(_depth_meters.at<float>(v,u) - projection.at<float>(v,u));
    if (projection.at<float>(v,u)<_depth_meters.at<float>(v,u) && diff>0.005 )
    {
      wrong_score+=std::min(1.0f,diff);
    }
  }
  avg_wrong_score = wrong_score/static_cast<float>(cloud->points.size());
}

class CompareWrongRatio
{
  public:
  bool operator()(const PoseHypo &p1, const PoseHypo &p2)
  {
    if (p1._wrong_ratio > p2._wrong_ratio) return true;
    return false;
  }
};



template<class PointT>
void PoseEstimator<PointT>::rejectByRender(float projection_thres, HandT42 *hand)
{
  printf("before projection check, #hypo=%d\n", _pose_hypos.size());
  const int H = 480, W = 640;
  std::vector<PoseHypo, Eigen::aligned_allocator<PoseHypo>> hypo_tmp = _pose_hypos;
  _pose_hypos.clear();
  PointCloud::Ptr cloud_tmp(new PointCloud);
  pcl::fromPCLPointCloud2(_obj_mesh->cloud, *cloud_tmp);
  PointCloudRGB::Ptr cloud_color(new PointCloudRGB);
  pcl::copyPointCloud(*cloud_tmp, *cloud_color);
  for (auto &pt : cloud_color->points)
  {
    pt.r = 0;
    pt.g = 0;
    pt.b = 255;
  }
  pcl::toPCLPointCloud2(*cloud_color, _obj_mesh->cloud);
  _renderer.clearScene();
  for (const auto &h : hand->_meshes)
  {
    std::string name = h.first;
    if (hand->_component_status[name] == false)
      continue;
    Eigen::Matrix4f tf_in_base;
    hand->getTFHandBase(name, tf_in_base);
    PointCloud::Ptr cloud_tmp(new PointCloud);
    pcl::fromPCLPointCloud2(h.second->cloud, *cloud_tmp);
    PointCloudRGB::Ptr cloud_color(new PointCloudRGB);
    pcl::copyPointCloud(*cloud_tmp, *cloud_color);
    for (auto &pt : cloud_color->points)
    {
      pt.r = 255;
      pt.g = 0;
      pt.b = 0;
    }
    pcl::toPCLPointCloud2(*cloud_color, h.second->cloud);
    _renderer.addObject(h.second, hand->_handbase_in_cam * tf_in_base);
  }

  std::vector<cv::Mat> depth_sims(hypo_tmp.size()), color_sims(hypo_tmp.size());
  for (int i = 0; i < hypo_tmp.size(); i++)
  {
    Eigen::Matrix4f model2scene = hypo_tmp[i]._pose;
    _renderer.addObject(_obj_mesh, model2scene);
    _renderer.doRender(Eigen::Matrix4d::Identity(), depth_sims[i], color_sims[i]);
    _renderer.removeLastObject();
  }

  const float roi_weight = cfg->yml["render_roi_weight"].as<float>();
#pragma omp parallel
  {
    cv::Mat depth_meters = _depth_meters.clone();

#pragma omp for schedule(dynamic)
    for (int i = 0; i < hypo_tmp.size(); i++)
    {
      Eigen::Matrix4f model2scene = hypo_tmp[i]._pose;

      //Compare
      float roi_diff = 0, bg_diff = 0;
      int roi_cnt = 0, bg_cnt = 0;
      // cv::Mat diff_map = cv::Mat::zeros(H,W,CV_32F);
      for (int h = 0; h < H; h++)
      {
        for (int w = 0; w < W; w++)
        {
          float sim = depth_sims[i].at<float>(h, w);
          float real = depth_meters.at<float>(h, w);
          float diff = 0;
          if ((real <= 0.1 || real >= 2.0) && (sim > 0.1 || sim < 2.0))
          {
            diff = 2.0;
          }
          else if ((sim <= 0.1 || sim >= 2.0) && (real > 0.1 || real < 2.0))
          {
            diff = 2.0;
          }
          else
          {
            diff = std::abs(sim - real);
          }

          if (color_sims[i].at<cv::Vec3b>(h, w)[0] == 255) //object part is blue
          {
            roi_diff += diff;
            roi_cnt++;
          }
          else
          {
            bg_diff += diff;
            bg_cnt++;
          }
        }
      }

      float diff_total = roi_weight * roi_diff / roi_cnt + bg_diff / bg_cnt;
      hypo_tmp[i]._wrong_ratio = diff_total;
    }
  }

  std::priority_queue<PoseHypo, std::vector<PoseHypo, Eigen::aligned_allocator<PoseHypo>>, CompareWrongRatio> Q;
  for (auto &p : hypo_tmp)
  {
    Q.push(p);
  }
  int num_to_keep = std::max(static_cast<int>(cfg->yml["render_keep_hypo"].as<float>() * hypo_tmp.size()), 10); // Top ratio to keep
  num_to_keep = std::min(num_to_keep, static_cast<int>(hypo_tmp.size()));
  for (int i = 0; i < num_to_keep; i++)
  {
    if (Q.empty())
      break;
    PoseHypo p = Q.top();
    Q.pop();
    _pose_hypos.push_back(p);
  }
  printf("after projection check, #hypo=%d\n", _pose_hypos.size());

}

template<class PointT>
void PoseEstimator<PointT>::selectBest(PoseHypo &best_hypo)
{
  best_hypo = _pose_hypos[0];
  float best_lcp = 0;

  const float lcp_dist = cfg->yml["lcp"]["dist"].as<float>();
  const float normal_angle = cfg->yml["lcp"]["normal_angle"].as<float>();

#pragma omp parallel
  {
    boost::shared_ptr<pcl::PointCloud<PointT>> model001(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*_model001, *model001);
    boost::shared_ptr<pcl::PointCloud<PointT>> scene_high_confidence(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*_scene_high_confidence,*scene_high_confidence);
    std::vector<float> scene_weights(_object_segment->points.size(), 1.0);

    #pragma omp for schedule(dynamic)
    for (int i = 0; i < _pose_hypos.size(); i++)
    {
      Eigen::Matrix4f model2scene = _pose_hypos[i]._pose;
      boost::shared_ptr<pcl::PointCloud<PointT>> transformed_model(new pcl::PointCloud<PointT>);
      pcl::transformPointCloudWithNormals(*model001, *transformed_model, model2scene);
      float lcp = Utils::computeLCP<PointT>(scene_high_confidence, scene_weights, transformed_model, lcp_dist, normal_angle, true, true, true);
      _pose_hypos[i]._lcp_score = lcp; // update lcp score

      #pragma omp critical
      if (_pose_hypos[i]._lcp_score > best_lcp)
      {
        best_hypo = _pose_hypos[i];
        best_lcp = lcp;
      }
    }
  }

  best_hypo.print();

}


template<class PointT>
void PoseEstimator<PointT>::registerMesh(std::string mesh_dir, std::string name, const Eigen::Matrix4f &pose)
{
  _sdf.registerMesh(mesh_dir, name, pose);
}

template<class PointT>
void PoseEstimator<PointT>::registerHandMesh(Hand *hand)
{
  for (const auto& h:hand->_convex_meshes)
  {
    if (!(h.first=="finger_1_1" || h.first=="finger_1_2" || h.first=="finger_2_1" || h.first=="finger_2_2")) continue;
    Eigen::Matrix4f model2handbase(Eigen::Matrix4f::Identity());
    hand->getTFHandBase(h.first, model2handbase);
    _sdf.registerMesh(h.second, h.first, model2handbase);
  }
}


template<class PointT>
void PoseEstimator<PointT>::rejectByCollisionOrNonTouching(HandT42* hand)
{
  if (cfg->yml["pose_estimator_use_physics"].as<bool>()==false)
  {
    printf("Not using physics\n");
    return;
  }
  std::vector<PoseHypo,Eigen::aligned_allocator<PoseHypo> > hypo_tmp = _pose_hypos;
  _pose_hypos.clear();

  std::vector<std::string> finger_names={"finger_1_2","finger_2_2", "finger_1_1", "finger_2_1"};
  std::map<std::string, Eigen::MatrixXf> finger_cloud_eigens;
  for (auto finger_name:finger_names)
  {
    if (hand->_component_status[finger_name]==false) continue;
    Eigen::Matrix4f finger2handbase(Eigen::Matrix4f::Identity());
    hand->getTFHandBase(finger_name,finger2handbase);
    PointCloudRGBNormal::Ptr finger(new PointCloudRGBNormal);
    pcl::transformPointCloudWithNormals( *(hand->_clouds[finger_name]), *finger, finger2handbase);
    // Utils::downsamplePointCloud(finger,finger,0.007);

    Eigen::MatrixXf cloud_eigen = finger->getMatrixXfMap();  // Mapping to orignal cloud data !
    Eigen::MatrixXf P = cloud_eigen.block(0,0,3,cloud_eigen.cols());
    P.transposeInPlace();
    finger_cloud_eigens[finger_name]=P;
  }

  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
  PointCloudRGBNormal::Ptr cloud_without_hand(new PointCloudRGBNormal);
  pcl::transformPointCloudWithNormals(*_cloud_withouthand_raw, *cloud_without_hand, hand->_handbase_in_cam.inverse());
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(cloud_without_hand, cloud_without_hand, 0.005);
  kdtree.setInputCloud (cloud_without_hand);

  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree_hand;  // In handbase frame
  kdtree_hand.setInputCloud(hand->_hand_cloud);

  // We compare in handbase frame
  const float collision_dist_ratio = cfg->yml["collision_thres"].as<float>();
  const float collision_dist = std::min(-_smallest_dim * collision_dist_ratio, -0.007f);
  const float inside_ob_dist = std::min(-_smallest_dim/5, -0.01f);
  const float non_touch_dist = cfg->yml["non_touch_dist"].as<float>();
  printf("collision_dist=%f, non_touch_dist=%f\n",collision_dist,non_touch_dist);
  const float collision_finger_dist = -cfg->yml["collision_finger_dist"].as<float>();
  Eigen::Matrix4f cam2handbase = hand->_handbase_in_cam.inverse();
  const float collision_finger_volume_ratio = cfg->yml["collision_finger_volume_ratio"].as<float>();

#pragma omp parallel
{
  SDFchecker sdf(_sdf);
  std::map<std::string, Eigen::MatrixXf> finger_cloud_eigens_local = finger_cloud_eigens;
  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree_local = kdtree;
  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree_hand_local = kdtree_hand;
  boost::shared_ptr<pcl::PointCloud<PointT>> model(new pcl::PointCloud<PointT>);
  pcl::copyPointCloud(*_model, *model);
  std::map<std::string, bool> hand_component_status = hand->_component_status;

  #pragma omp for schedule(dynamic)
  for (int i=0;i<hypo_tmp.size();i++)
  {
    const int id=hypo_tmp[i]._id;
    bool rejected=false;
    Eigen::Matrix4f model2scene = hypo_tmp[i]._pose;
    Eigen::Matrix4f model2handbase =  cam2handbase * model2scene;
    sdf.transformMesh("object",model2handbase);  //NOTE: we are doing all this in handbase frame
    Eigen::Vector4f model_center_init_eigen(_model_center_init.x, _model_center_init.y, _model_center_init.z, 1.0);
    Eigen::Vector4f cur_center_eigen = model2handbase * model_center_init_eigen;
    pcl::PointXYZRGBNormal cur_center;    //NOTE: in handbase frame
    cur_center.x = cur_center_eigen(0);
    cur_center.y = cur_center_eigen(1);
    cur_center.z = cur_center_eigen(2);
    std::vector<int> indices;
    std::vector<float> sq_distances;

    // Check if any scene point is inside current model pose, reject it
    int kdtree_res = kdtree_local.nearestKSearch(cur_center, 1, indices, sq_distances);  //Check nearest point is enough
    if (kdtree_res>0)
    {
      Eigen::MatrixXf near_pts(1,3);
      auto pt = cloud_without_hand->points[indices[0]];
      near_pts(0,0) = pt.x;
      near_pts(0,1) = pt.y;
      near_pts(0,2) = pt.z;

      float min_dist=std::numeric_limits<float>::max(), max_dist=-std::numeric_limits<float>::max();
      int num_inner_pt=0;
      float lower_bound=-std::numeric_limits<float>::max(), upper_bound=std::numeric_limits<float>::max();
      std::vector<float> dists;
      sdf.getSignedDistanceMinMaxWithRegistered("object", near_pts,lower_bound,upper_bound,min_dist,max_dist, dists, num_inner_pt);
      if (min_dist<=inside_ob_dist)
      {
        sdf.transformMesh("object",model2handbase.inverse());
        continue;
      }
    }

    // One round of Quick check: point from hand that is nearest to object center
    if (kdtree_hand_local.nearestKSearch(cur_center, 1, indices, sq_distances)>0)
    {
      if (std::sqrt(sq_distances[0]) < _ob_diameter/2)
      {
        auto nei = hand->_hand_cloud->points[indices[0]];
        Eigen::MatrixXf near_pts(1,3);
        near_pts(0,0) = nei.x;
        near_pts(0,1) = nei.y;
        near_pts(0,2) = nei.z;
        float min_dist=std::numeric_limits<float>::max(), max_dist=-std::numeric_limits<float>::max();
        int num_inner_pt=0;
        float lower_bound=-std::numeric_limits<float>::max(), upper_bound=std::numeric_limits<float>::max();
        std::vector<float> dists;
        sdf.getSignedDistanceMinMaxWithRegistered("object", near_pts,lower_bound,upper_bound,min_dist,max_dist, dists, num_inner_pt);
        if (min_dist<collision_dist)
        {
          sdf.transformMesh("object",model2handbase.inverse());
          continue;
        }

      }
    }


    // Check hand collision inside object
    std::map<std::string, bool> non_touch;
    for (const auto &h:finger_cloud_eigens_local)
    {
      std::string name=h.first;
      if (hand_component_status["finger_1_1"]==false && (name=="finger_1_1" || name=="finger_1_2")) continue;
      if (hand_component_status["finger_2_1"]==false && (name=="finger_2_1" || name=="finger_2_2")) continue;
      float min_dist=std::numeric_limits<float>::max(), max_dist=-std::numeric_limits<float>::max();
      int num_inner_pt=0;
      float lower_bound=-std::numeric_limits<float>::max(), upper_bound=std::numeric_limits<float>::max();
      std::vector<float> dists;
      sdf.getSignedDistanceMinMaxWithRegistered("object", h.second,lower_bound,upper_bound,min_dist, max_dist, dists, num_inner_pt);
      if (min_dist<=collision_dist)   //if end finger status false(0 deg), still check collision
      {
        rejected=true;
        break;
      }
      // No collision, then check if touching is possible
      if (min_dist>non_touch_dist && hand_component_status[name])
      {
        non_touch[name]=true;
      }
    }
    if (rejected)
    {
      sdf.transformMesh("object",model2handbase.inverse());
      continue;
    }

    //One side not touch
    if ( (non_touch["finger_1_1"] && non_touch["finger_1_2"]) || (non_touch["finger_2_1"] && non_touch["finger_2_2"]) )
    {
      sdf.transformMesh("object",model2handbase.inverse());
      continue;
    }

    // Check object's penetration inside finger
    boost::shared_ptr<pcl::PointCloud<PointT>> model_in_handbase(new pcl::PointCloud<PointT>);
    pcl::transformPointCloudWithNormals(*model, *model_in_handbase, model2handbase);
    Eigen::MatrixXf model_eigen = model_in_handbase->getMatrixXfMap();  // Mapping to orignal cloud data !
    Eigen::MatrixXf P = model_eigen.block(0,0,3,model_eigen.cols());
    P.transposeInPlace();
    for (const auto &h:hand_component_status)
    {
      if (h.first.find("finger")==-1) continue;
      std::string finger_name = h.first;
      // pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree_fingers(*hand->_kdtrees[finger_name]);
      // std::vector<int> indices;
      // std::vector<float> sq_dists;
      // if (kdtree_local.nearestKSearch(cur_center, 1, indices, sq_dists)<=0) continue;
      // if (std::sqrt(sq_dists[0]) >= ob_diameter/2) continue;

      float min_dist=std::numeric_limits<float>::max(), max_dist=-std::numeric_limits<float>::max();
      int num_inner_pt=0;
      float lower_bound=-std::numeric_limits<float>::max(), upper_bound=std::numeric_limits<float>::max();
      std::vector<float> dists;
      sdf.getSignedDistanceMinMaxWithRegistered(finger_name, P, lower_bound,upper_bound,min_dist, max_dist, dists, num_inner_pt);
      if (min_dist<collision_finger_dist)
      {
        rejected = true;
        break;
      }
      int num_inside = 0;
      for (const auto &d:dists)
      {
        if (d<0)
        {
          num_inside++;
        }
      }
      if (num_inside/P.rows() > collision_finger_volume_ratio)
      {
        rejected = true;
        break;
      }

    }
    if (rejected)
    {
      sdf.transformMesh("object",model2handbase.inverse());
      continue;
    }

    sdf.transformMesh("object",model2handbase.inverse());

    #pragma omp critical
    _pose_hypos.push_back(hypo_tmp[i]);
  }
}

}
template class PoseEstimator<pcl::PointSurfel>;