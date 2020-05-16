#include "Utils.h"
#include "ConfigParser.h"
#include "Renderer.h"
#include "Hand.h"
#include "PoseEstimator.h"
#include "tinyxml2.h"
#include "yaml-cpp/yaml.h"
#include "PoseHypo.h"


using namespace Eigen;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "main_real_auto");
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  std::string config_dir;
  if (argc<2)
  {
    config_dir = "/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/github/icra20-hand-object-pose/config_autodataset.yaml";
  }
  else
  {
    config_dir = std::string(argv[1]);
  }
  std::cout<<"Using config file: "<<config_dir<<std::endl;
  ConfigParser cfg(config_dir);
  std::map<std::vector<int>, std::vector<std::pair<int, int>>> ppfs;
  std::ifstream ppf_file(cfg.yml["ppf_path"].as<std::string>(), std::ios::binary);
  boost::archive::binary_iarchive iarch(ppf_file);
  iarch >> ppfs;

  PointCloudSurfel::Ptr model(new PointCloudSurfel);
  pcl::io::loadPLYFile(cfg.object_model_path, *model);
  assert(model->points.size()>0);
  PointCloudSurfel::Ptr model001(new PointCloudSurfel);
  Utils::downsamplePointCloud(model, model001, 0.001);
  Utils::downsamplePointCloud(model001, model, 0.005);

  // Compute min gripper dist
  {
    pcl::PointSurfel minPt, maxPt;
    pcl::getMinMax3D(*model001, minPt, maxPt);
    cfg.gripper_min_dist = 0.8 * std::min(std::min(std::abs(minPt.x - maxPt.x), std::abs(minPt.y - maxPt.y)), std::abs(minPt.z - maxPt.z));
  }

  // We treat Motoman left arm as world !!!!!
  Eigen::Matrix4f base_in_world = cfg.leftarm_in_base.inverse() * cfg.palm_in_baselink * cfg.handbase_in_palm;
  Eigen::Matrix4f handbase_in_leftarm = base_in_world;

  // handbase in left cam
  Eigen::Matrix4f handbase_in_cam = cfg.cam1_in_leftarm.inverse() * handbase_in_leftarm;

  cv::Mat scene_depth;
  Utils::readDepthImage(scene_depth, cfg.depth_path);
  cv::Mat scene_bgr = cv::imread(cfg.rgb_path);

  PointCloudRGBNormal::Ptr scene_rgb(new PointCloudRGBNormal);
  Utils::convert3dOrganizedRGB<pcl::PointXYZRGBNormal>(scene_depth, scene_bgr, cfg.cam_intrinsic, scene_rgb);

  Utils::calNormalIntegralImage<pcl::PointXYZRGBNormal>(scene_rgb, -1, 0.02, 10, true); // method 0 for curvature computation

  {
    pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
    pass.setInputCloud(scene_rgb);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 2.0);
    pass.filter(*scene_rgb);
  }

  PointCloudRGBNormal::Ptr scene_organized(new PointCloudRGBNormal);
  pcl::copyPointCloud(*scene_rgb, *scene_organized);
  Utils::downsamplePointCloud<pcl::PointXYZRGBNormal>(scene_rgb, scene_rgb, 0.001);

  Eigen::Matrix4f cam_in_handbase = handbase_in_cam.inverse();
  pcl::transformPointCloudWithNormals(*scene_rgb, *scene_rgb, cam_in_handbase);

  {
    pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
    pass.setInputCloud(scene_rgb);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.12, 0.05);
    pass.filter(*scene_rgb);

    pass.setInputCloud(scene_rgb);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.25, -0.07);
    pass.filter(*scene_rgb);

    pass.setInputCloud(scene_rgb);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.2, 0.2);
    pass.filter(*scene_rgb);
  }

  pcl::transformPointCloudWithNormals(*scene_rgb, *scene_rgb, cam_in_handbase.inverse());


  HandT42 hand(&cfg, cfg.cam_intrinsic);
  hand.setCurScene(scene_depth, scene_organized, scene_rgb, handbase_in_cam);
  // hand.printComponents();

  hand.makeHandCloud();

  bool match1 = false;
  bool match2 = false;
  const float finger1_min_match = cfg.yml["hand_match"]["finger1_min_match"].as<float>();
  const float finger2_min_match = cfg.yml["hand_match"]["finger2_min_match"].as<float>();
  const float finger1_dist_thres = cfg.yml["hand_match"]["finger1_dist_thres"].as<float>();
  const float finger2_dist_thres = cfg.yml["hand_match"]["finger2_dist_thres"].as<float>();
  const float finger1_normal_angle = cfg.yml["hand_match"]["finger1_normal_angle"].as<float>();
  const float finger2_normal_angle = cfg.yml["hand_match"]["finger2_normal_angle"].as<float>();

  if (cam_in_handbase(1, 3) > 0) // Cam on the right side of hand
  {
    match1 = hand.matchOneComponentPSO("finger_2_1", 0, 120, false, finger1_dist_thres, finger1_normal_angle, finger1_min_match);
    if (match1)
    {
      hand.matchOneComponentPSO("finger_2_2", 0, 90, true, finger2_dist_thres, finger2_normal_angle, finger2_min_match);
    }
    match2 = hand.matchOneComponentPSO("finger_1_1", 0, 120, false, finger1_dist_thres, finger1_normal_angle, finger1_min_match);
    if (match2)
    {
      hand.matchOneComponentPSO("finger_1_2", 0, 90, true, finger2_dist_thres, finger2_normal_angle, finger2_min_match);
    }
  }
  else
  {
    match2 = hand.matchOneComponentPSO("finger_1_1", 0, 120, false, finger1_dist_thres, finger1_normal_angle, finger1_min_match);
    if (match2)
    {
      hand.matchOneComponentPSO("finger_1_2", 0, 90, true, finger2_dist_thres, finger2_normal_angle, finger2_min_match);
    }
    match1 = hand.matchOneComponentPSO("finger_2_1", 0, 120, false, finger1_dist_thres, finger1_normal_angle, finger1_min_match);
    if (match1)
    {
      hand.matchOneComponentPSO("finger_2_2", 0, 90, true, finger2_dist_thres, finger2_normal_angle, finger2_min_match);
    }
  }

  hand.adjustHandHeight();
  hand.makeHandCloud();

  PointCloudSurfel::Ptr object1(new PointCloudSurfel);
  PointCloudSurfel::Ptr tmp_scene(new PointCloudSurfel);
  pcl::copyPointCloud(*scene_rgb, *tmp_scene);
  const float near_dist = cfg.yml["near_hand_dist"].as<float>();
  hand.removeSurroundingPointsAndAssignProbability<pcl::PointSurfel, true>(tmp_scene, object1, near_dist * near_dist);

  PointCloudRGBNormal::Ptr cloud_withouthand_raw(new PointCloudRGBNormal);
  pcl::copyPointCloud(*object1, *cloud_withouthand_raw);

  // pcl::io::savePLYFile("/home/bowen/debug/scene_after_hand_dense.ply", *object1);
  Utils::calNormalMLS<pcl::PointSurfel>(object1, 0.003);

  PointCloudSurfel::Ptr object_segment(new PointCloudSurfel);
  pcl::copyPointCloud(*object1, *object_segment);
  Utils::downsamplePointCloud<pcl::PointSurfel>(object_segment, object_segment, 0.003);
  Utils::removeAllNaNFromPointCloud(object_segment);
  for (auto &pt : object_segment->points)
  {
    pcl::flipNormalTowardsViewpoint(pt, 0, 0, 0, pt.normal[0], pt.normal[1], pt.normal[2]);
  }

  pcl::KdTreeFLANN<pcl::PointSurfel> kdtree;
  kdtree.setInputCloud(object1);
  for (auto &pt : object_segment->points)
  {
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    if (kdtree.nearestKSearch(pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0)
    {
      pt.confidence = 0;
      continue;
    }
    pt.confidence = object1->points[pointIdxNKNSearch[0]].confidence;
  }

  PointCloudSurfel::Ptr scene_003(new PointCloudSurfel);
  pcl::copyPointCloud(*scene_rgb, *scene_003);
  Utils::downsamplePointCloud<pcl::PointSurfel>(scene_003, scene_003, 0.003);

  PoseEstimator<pcl::PointSurfel> est(&cfg, model, model001, cfg.cam_intrinsic);
  est.setCurScene(scene_003, cloud_withouthand_raw, object_segment, scene_bgr, scene_depth);
  est.registerHandMesh(&hand);
  est.registerMesh(cfg.object_mesh_path, "object", Eigen::Matrix4f::Identity());
  bool succeed = est.runSuper4pcs(ppfs);

  if (!succeed)
  {
    printf("No pose found...\n");
    std::ofstream ff("/home/bowen/debug/model2scene.txt");
    ff << Eigen::Matrix4f::Identity() << std::endl;
    ff.close();
    exit(1);
  }

  est.clusterPoses(30, 0.015, true);
  est.refineByICP();
  est.clusterPoses(5, 0.003, false);
  est.rejectByCollisionOrNonTouching(&hand);
  est.rejectByRender(cfg.pose_estimator_wrong_ratio, &hand);
  PoseHypo best(-1);
  est.selectBest(best);
  Eigen::Matrix4f model2scene = best._pose;
  std::cout << "best tf:\n"
            << model2scene << "\n\n";

  const std::string out_dir = cfg.yml["out_dir"].as<std::string>();
  pcl::PolygonMesh::Ptr model_viz(new pcl::PolygonMesh);
  pcl::io::loadOBJFile(cfg.object_mesh_path, *model_viz);
  Utils::transformPolygonMesh(model_viz, model_viz, model2scene);
  pcl::io::saveOBJFile(out_dir+"/best.obj", *model_viz);
  pcl::io::savePLYFile(out_dir+"/scene_normals.ply", *scene_rgb);
  PointCloudRGBNormal::Ptr hand_cloud(new PointCloudRGBNormal);
  pcl::transformPointCloudWithNormals(*hand._hand_cloud, *hand_cloud, hand._handbase_in_cam);
  pcl::io::savePLYFile(out_dir+"/hand.ply", *hand_cloud);

  std::ofstream ff(out_dir+"/model2scene.txt");
  ff << model2scene << std::endl;
  ff.close();
}