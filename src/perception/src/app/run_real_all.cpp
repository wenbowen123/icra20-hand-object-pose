#include "Utils.h"
#include "ConfigParser.h"
#include "Renderer.h"
#include "Hand.h"
#include "PoseEstimator.h"
#include "tinyxml2.h"
#include "yaml-cpp/yaml.h"
#include "PoseHypo.h"

using namespace Eigen;

cv::Mat rgb_image, depth_meters;
Eigen::Matrix4f palm_in_baselink(Eigen::Matrix4f::Identity()), leftarm_in_base(Eigen::Matrix4f::Identity());

int main(int argc, char **argv)
{
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  ros::init(argc, argv, "run_real_all");
  std::string config_dir;
  if (argc>=2)
  {
    config_dir = std::string(argv[1]);
    std::cout<<"Reading from config: "<<config_dir<<std::endl;
  }
  else
  {
    config_dir = "/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/catkin_ws/src/icra20_manipulation_pose/config_autodataset.yaml";
  }
  ConfigParser cfg(config_dir);

  const std::string base_dir = "/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/catkin_ws/src/icra20_manipulation_pose/motoman_data/auto_collect/";
  const std::string model_name = cfg.yml["model_name"].as<std::string>();
  std::vector<std::string> record_folders;
  Utils::readDirectory(base_dir+model_name+"/", record_folders);

  const float finger1_min_match = cfg.yml["hand_match"]["finger1_min_match"].as<float>();
  const float finger2_min_match = cfg.yml["hand_match"]["finger2_min_match"].as<float>();
  const float finger1_dist_thres = cfg.yml["hand_match"]["finger1_dist_thres"].as<float>();
  const float finger2_dist_thres = cfg.yml["hand_match"]["finger2_dist_thres"].as<float>();
  const float finger1_normal_angle = cfg.yml["hand_match"]["finger1_normal_angle"].as<float>();
  const float finger2_normal_angle = cfg.yml["hand_match"]["finger2_normal_angle"].as<float>();

  std::map<std::vector<int>, std::vector<std::pair<int, int>>> ppfs;
  std::ifstream ppf_file(cfg.yml["ppf_path"].as<std::string>(), std::ios::binary);
  boost::archive::binary_iarchive iarch(ppf_file);
  iarch >> ppfs;

  PointCloudSurfel::Ptr model(new PointCloudSurfel);
  pcl::io::loadPLYFile(cfg.object_model_path, *model);
  PointCloudSurfel::Ptr model001(new PointCloudSurfel);
  Utils::downsamplePointCloud(model, model001, 0.001);
  Utils::downsamplePointCloud(model001, model, 0.005);

  // Compute min gripper dist
  {
    pcl::PointSurfel minPt, maxPt;
    pcl::getMinMax3D(*model, minPt, maxPt);
    cfg.gripper_min_dist = 0.8*std::min(std::min(std::abs(minPt.x - maxPt.x), std::abs(minPt.y - maxPt.y)), std::abs(minPt.z - maxPt.z));
    printf("gripper_min_dist is set to %f\n", cfg.gripper_min_dist);
  }

  PointCloudRGBNormal::Ptr scene_organized(new PointCloudRGBNormal);
  PointCloudRGBNormal::Ptr scene_rgb(new PointCloudRGBNormal);
  HandT42 hand(&cfg,cfg.cam_intrinsic);

  PointCloudSurfel::Ptr scene_003(new PointCloudSurfel);
  PointCloudRGBNormal::Ptr cloud_withouthand_raw(new PointCloudRGBNormal);
  PointCloudSurfel::Ptr object_segment(new PointCloudSurfel);
  PoseEstimator<pcl::PointSurfel> est(&cfg, model, model001, cfg.cam_intrinsic);
  for (const auto &record_folder:record_folders)
  {
    const std::string record_dir = base_dir+model_name+"/"+record_folder+"/";
    std::vector<std::string> tmp_files;
    Utils::readDirectory(record_dir, tmp_files);
    std::vector<std::string> color_files;
    for (const auto &file:tmp_files)
    {
      if (file.find("rgb")!=-1)
      {
        color_files.push_back(record_dir+file);
      }
    }
    std::sort(color_files.begin(),color_files.end());
    for (const auto &color_file:color_files)
    {
      std::cout<<">>>>>>>>>>>>>>>>>> processing "+color_file<<std::endl;
      system(std::string("rm -rf  /home/bowen/debug/model2scene.txt ").c_str());
      system(std::string("rm -rf  /home/bowen/debug/scene_normals.ply ").c_str());
      system(std::string("rm -rf  /home/bowen/debug/scene_after_hand.ply ").c_str());
      system(std::string("rm -rf  /home/bowen/debug/hand.ply ").c_str());
      system(std::string("rm -rf  /home/bowen/debug/best.obj ").c_str());
      int pos1 = color_file.find("rgb");
      int pos2 = color_file.find(".");
      int index = std::stoi(color_file.substr(pos1+3,pos2-pos1-3));
      printf("index=%d\n",index);
      const std::string leftarm_in_base_file = record_dir + "arm_left_link_7_t_"+std::to_string(index)+".txt";
      leftarm_in_base.setIdentity();
      Utils::parsePoseTxt(leftarm_in_base_file, leftarm_in_base);

      const std::string palm_in_baselink_file = record_dir + "palm_in_base"+std::to_string(index)+".txt";
      palm_in_baselink.setIdentity();
      Utils::parsePoseTxt(palm_in_baselink_file, palm_in_baselink);

      const std::string depth_file = record_dir + "depth"+std::to_string(index)+".png";
      Utils::readDepthImage(depth_meters, depth_file);
      rgb_image = cv::imread(color_file);


      Eigen::Matrix4f handbase_in_leftarm = leftarm_in_base.inverse() * palm_in_baselink * cfg.handbase_in_palm;
      Eigen::Matrix4f handbase_in_cam = cfg.cam1_in_leftarm.inverse() * handbase_in_leftarm;

      PointCloudRGBNormal::Ptr scene_rgb(new PointCloudRGBNormal);
      Utils::convert3dOrganizedRGB<pcl::PointXYZRGBNormal>(depth_meters, rgb_image, cfg.cam_intrinsic, scene_rgb);
      assert(scene_rgb->points.size()>0);
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
      pcl::io::savePLYFile("/home/bowen/debug/scene_normals.ply",*scene_rgb);

      Eigen::Matrix4f cam_in_handbase = handbase_in_cam.inverse();
      pcl::transformPointCloudWithNormals(*scene_rgb, *scene_rgb, cam_in_handbase);

      //Crop hand region based on hand dimensions
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
      std::cout << "handbase_in_cam:\n"
                << handbase_in_cam << "\n\n";

      hand.setCurScene(depth_meters, scene_organized, scene_rgb, handbase_in_cam);
      hand.makeHandCloud();

      bool match1 = false;
      bool match2 = false;
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
      PointCloudRGBNormal::Ptr hand_cloud(new PointCloudRGBNormal);
      pcl::transformPointCloudWithNormals(*(hand._hand_cloud), *hand_cloud, hand._handbase_in_cam);
      pcl::io::savePLYFile("/home/bowen/debug/hand.ply",*hand_cloud);

      PointCloudSurfel::Ptr object1(new PointCloudSurfel);
      PointCloudSurfel::Ptr tmp_scene(new PointCloudSurfel);
      pcl::copyPointCloud(*scene_rgb, *tmp_scene);
      const float near_dist = cfg.yml["near_hand_dist"].as<float>();
      hand.removeSurroundingPointsAndAssignProbability<pcl::PointSurfel, true>(tmp_scene, object1, near_dist * near_dist);

      pcl::copyPointCloud(*object1, *cloud_withouthand_raw);

      Utils::calNormalMLS<pcl::PointSurfel>(object1, 0.003);

      pcl::copyPointCloud(*object1, *object_segment);
      Utils::downsamplePointCloud<pcl::PointSurfel>(object_segment, object_segment, 0.003);
      Utils::removeAllNaNFromPointCloud(object_segment);
      for (auto &pt : object_segment->points)
      {
        pcl::flipNormalTowardsViewpoint(pt, 0, 0, 0, pt.normal[0], pt.normal[1], pt.normal[2]);
      }
      pcl::io::savePLYFile("/home/bowen/debug/scene_after_hand.ply", *object_segment);
      std::cout << "normals compute done\n";
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

      // Perform alignment
      pcl::copyPointCloud(*scene_rgb, *scene_003);
      Utils::downsamplePointCloud<pcl::PointSurfel>(scene_003, scene_003, 0.003);
      pcl::io::savePLYFile("/home/bowen/debug/scene_003.ply", *scene_003);
      est.setCurScene(scene_003, cloud_withouthand_raw, object_segment, rgb_image, depth_meters);
      est.registerHandMesh(&hand);
      est.registerMesh(cfg.object_mesh_path, "object", Eigen::Matrix4f::Identity());
      bool succeed = est.runSuper4pcs(ppfs);
      assert(succeed);
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

      pcl::PolygonMesh::Ptr model_viz(new pcl::PolygonMesh);
      pcl::io::loadOBJFile(cfg.object_mesh_path, *model_viz);
      Utils::transformPolygonMesh(model_viz, model_viz, model2scene);
      pcl::io::saveOBJFile("/home/bowen/debug/best.obj", *model_viz);

      std::ofstream ff("/home/bowen/debug/model2scene.txt");
      ff << model2scene << std::endl;
      ff.close();


      const std::string pred_dir = record_dir + "predict/"+std::to_string(index)+"/";
      system(std::string("mkdir -p "+pred_dir).c_str());
      system(std::string("cp /home/bowen/debug/model2scene.txt "+pred_dir+"model2scene.txt").c_str());
      system(std::string("cp /home/bowen/debug/scene_normals.ply "+pred_dir+"scene_normals.ply").c_str());
      system(std::string("cp /home/bowen/debug/scene_after_hand.ply "+pred_dir+"scene_after_hand.ply").c_str());
      system(std::string("cp /home/bowen/debug/hand.ply "+pred_dir+"hand.ply").c_str());
      system(std::string("cp /home/bowen/debug/best.obj "+pred_dir+"best.obj").c_str());
      std::cout<<pred_dir<<std::endl;

      hand.reset();
      est.reset();

      rgb_image.release();
      depth_meters.release();
      // exit(1);
    }

  }

}
