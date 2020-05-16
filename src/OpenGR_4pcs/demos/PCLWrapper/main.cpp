#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
// Basic PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>

// For IO
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/registration/transformation_estimation_dual_quaternion.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/super4pcs.h>
#include <pcl/filters/extract_indices.h>

#include <gr/shared.h>
#include "../demo-utils.h"

//The PCL wrapper source code is available at install_dir/include/pcl/registration

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudNormal;

//@cloud1: source
//@cloud2: target
template<class PointType>
bool runICP(boost::shared_ptr<pcl::PointCloud<PointType>> cloud1, boost::shared_ptr<pcl::PointCloud<PointType>> cloud2, Eigen::Matrix4f &trans)
{
  // boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointType, PointType, float> >
  //     te (new pcl::registration::TransformationEstimationLM<PointType, PointType, float>);
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  // icp.setTransformationEstimation (te);
  icp.setUseReciprocalCorrespondences(true);
  icp.setMaximumIterations(100);
  icp.setRANSACIterations(100);
  icp.setMaxCorrespondenceDistance(100);
  icp.setTransformationEpsilon (1e-9);
  icp.setInputSource(cloud1);
  icp.setInputTarget(cloud2);
  pcl::PointCloud<PointType> Final;
  icp.align(Final);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore() << std::endl;
  trans = icp.getFinalTransformation();

  return icp.hasConverged();
}

//@cloud1: source
//@cloud2: target
template<class PointType>
void runICP(boost::shared_ptr<pcl::PointCloud<PointType>> pclSegment,
                     boost::shared_ptr<pcl::PointCloud<PointType>> pclModel,
                     Eigen::Matrix4f &offsetTransform)
{

  PointCloudNormal::Ptr modelCloud(new PointCloudNormal);
  PointCloudNormal::Ptr segmentCloud(new PointCloudNormal);
  PointCloudNormal segCloudTrans;
  pcl::copyPointCloud(*pclModel, *modelCloud);
  pcl::copyPointCloud(*pclSegment, *segmentCloud);

  std::vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*modelCloud, *modelCloud, indices);
  pcl::removeNaNNormalsFromPointCloud(*segmentCloud, *segmentCloud, indices);

  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>::Ptr icp(new pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>());
  icp->setUseReciprocalCorrespondences(true);
  icp->setMaxCorrespondenceDistance(0.01);
  icp->setMaximumIterations(100);
  icp->setRANSACIterations(100);
  icp->setInputSource(segmentCloud); // not cloud_source, but cloud_source_trans!
  icp->setInputTarget(modelCloud);

  icp->align(segCloudTrans);
  if (icp->hasConverged())
  {
    offsetTransform = icp->getFinalTransformation();
    std::cout << "ICP score: " << icp->getFitnessScore() << std::endl;
  }
  else
  {
    std::cout << "ICP did not converge." << std::endl;
    offsetTransform << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
  }
}

using PointType=PointNT;
template<class PointType>
void calNormalMLS(boost::shared_ptr<pcl::PointCloud<PointType>> cloud, float normal_radius)
{
  pcl::MovingLeastSquares<PointType, PointType> mls;
  mls.setComputeNormals (true);
  mls.setPolynomialFit (true);
  mls.setNumberOfThreads(14);
  mls.setPolynomialOrder(2);
  mls.setInputCloud (cloud);
  boost::shared_ptr<pcl::search::KdTree<PointType> > tree (new pcl::search::KdTree<PointType>);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (normal_radius);
  boost::shared_ptr<pcl::PointCloud<PointType>> cloud_normal(new pcl::PointCloud<PointType>);
  mls.process (*cloud_normal);
  pcl::PointIndicesPtr indices = mls.getCorrespondingIndices ();
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (indices);
  extract.setNegative (false);
  extract.filter (*cloud);
  pcl::concatenateFields (*cloud, *cloud_normal, *cloud);
}

using namespace gr;

// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
  // Point clouds
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr aligned (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);

  // Get input object and scene
  if (argc < 4)
  {
    pcl::console::print_error ("Syntax is: %s OBJFILEscene OBJFILEobject [PARAMS]\n", argv[0]);
    Demo::printParameterList();
    return (-1);
  }

  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadOBJFile<PointNT> (argv[2], *object) < 0 ||
      pcl::io::loadOBJFile<PointNT> (argv[1], *scene) < 0)
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (-1);
  }

  if(int c = Demo::getArgs(argc, argv) != 0)
    {
      Demo::printUsage(argc, argv);
      exit(std::max(c,0));
    }




  pcl::Super4PCS<PointNT,PointNT> align;
  Demo::setOptionsFromArgs(align.options_);

  // Downsample
  pcl::console::print_highlight("Downsampling...\n");
  pcl::VoxelGrid<PointNT> grid;
  const float leaf = 0.005f;
  grid.setLeafSize(leaf, leaf, leaf);
  grid.setInputCloud(object);
  grid.filter(*object);
  grid.setInputCloud(scene);
  grid.filter(*scene);

  calNormalMLS(object, 0.01);
  calNormalMLS(scene, 0.01);
  for (auto &pt:scene->points)
  {
    pcl::flipNormalTowardsViewpoint (pt, 0,0,0,
          pt.normal[0],
          pt.normal[1],
          pt.normal[2]);
  }
  for (auto &pt:object->points)
  {
    pcl::flipNormalTowardsViewpoint (pt, 0, 0, 0,
          pt.normal[0],
          pt.normal[1],
          pt.normal[2]);
    pt.normal[0] = -pt.normal[0];
    pt.normal[1] = -pt.normal[1];
    pt.normal[2] = -pt.normal[2];
  }
  std::cout<<"normals compute done\n";

  // Perform alignment
  pcl::console::print_highlight ("Starting alignment...\n");
  align.setInputSource (scene);
  align.setInputTarget (object);

  {
    pcl::ScopeTime t("Alignment");
    align.align (*aligned);
  }


  Eigen::Matrix4f scene2model;
  Eigen::Matrix4f model2scene;
  model2scene.setIdentity();
  if (align.hasConverged ())
  {
    // Print results
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();   // scene2model
    scene2model = transformation;

    PointCloudT::Ptr tmp(new PointCloudT);
    pcl::copyPointCloud(*object, *tmp);
    pcl::transformPointCloudWithNormals (*tmp, *tmp, transformation.inverse());
    pcl::io::savePLYFile("/home/bowen/debug/4pcs_ob_in_scene_beforeicp.ply",*tmp);


    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    runICP<PointNT>(aligned,object,transformation);
    pcl::transformPointCloudWithNormals (*aligned, *aligned, transformation);
    scene2model = transformation * scene2model;

    model2scene = scene2model.inverse();
    pcl::transformPointCloudWithNormals (*object, *object, model2scene);
    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment - Super4PCS");
    visu.addPointCloud (object, ColorHandlerT (object, 0.0, 255.0, 0.0), "model");
    visu.addPointCloud (aligned, ColorHandlerT (aligned, 0.0, 0.0, 255.0), "aligned");
    visu.spin ();
  }
  else
  {
    pcl::console::print_error ("Alignment failed!\n");
    return (-1);
  }
  pcl::io::savePLYFile("/home/bowen/debug/4pcs_scene_tf.ply",*aligned);
  pcl::io::savePLYFile("/home/bowen/debug/4pcs_model2scene.ply",*object);
  pcl::io::savePLYFile("/home/bowen/debug/scene_normals.ply",*scene);


  return (0);
}
