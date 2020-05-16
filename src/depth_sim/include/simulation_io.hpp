#ifndef PCL_SIMULATION_IO_
#define PCL_SIMULATION_IO_

#include <boost/shared_ptr.hpp>

#include <GL/glew.h>

#include <pcl/pcl_config.h>
#ifdef OPENGL_IS_A_FRAMEWORK
# include <OpenGL/gl.h>
# include <OpenGL/glu.h>
#else
# include <GL/gl.h>
# include <GL/glu.h>
#endif
#ifdef GLUT_IS_A_FRAMEWORK
# include <GLUT/glut.h>
#else
# include <GL/glut.h>
#endif

// define the following in order to eliminate the deprecated headers warning
#define VTK_EXCLUDE_STRSTREAM_HEADERS
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/vtk_lib_io.h>


#include <camera.h>
#include <scene.h>
#include <range_likelihood.h>

#include <opencv/cv.h>

#include <vector>

namespace pcl
{
  namespace simulation
  {
    class PCL_EXPORTS SimExample
    {
      public:
        typedef boost::shared_ptr<SimExample> Ptr;
        typedef boost::shared_ptr<const SimExample> ConstPtr;
    	
        SimExample (int argc, char** argv,
    		int height,int width,float fx, float fy, float cx, float cy);
        void initializeGL (int argc, char** argv);
        void reset();
        
        void doSim (Eigen::Isometry3d pose_in);
    
        void write_score_image(const float* score_buffer,std::string fname);
        void write_depth_image(const float* depth_buffer,std::string fname);
        void write_depth_image_uint(const float* depth_buffer,std::string fname);
        void write_rgb_image(const uint8_t* rgb_buffer,std::string fname);
        void get_rgb_image_cv(const uint8_t *rgb_buffer, cv::Mat& bgr);

        void get_depth_image_uint(const float* depth_buffer, std::vector<unsigned short>* depth_img_uint);
        void get_depth_image_cv(const float* depth_buffer, cv::Mat &depth_image);

      public:
        Scene::Ptr scene_;
        Camera::Ptr camera_;
        RangeLikelihood::Ptr rl_;  
    
      private:
        uint16_t t_gamma[2048];  
    
        // of platter, usually 640x480
        int width_;
        int height_;
    };
  }
}




#endif
