/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_REGISTRATION_OPENGR_HPP_
#define PCL_REGISTRATION_OPENGR_HPP_

#include <pcl/io/ply_io.h>

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed()  {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void print(std::string message = ""){
        double t = elapsed();
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl {
template <typename PointSource, typename PointTarget> void
pcl::Super4PCS<PointSource, PointTarget>::computeTransformation (PointCloudSource &output, const Eigen::Matrix4f& guess)
{
  using namespace gr;

  // Initialize results
  final_transformation_ = guess;

  constexpr ::gr::Utils::LogLevel loglvl = ::gr::Utils::NoLog;

  gr::Utils::Logger logger(loglvl);
  MatcherType matcher(options_, logger);   // inherited from MatchBase
  matcher._ppfs = _ppfs;

  SamplerType sampler;
  TransformVisitor visitor;

  std::vector<gr::Point3D> set1, set2;

  // init Super4PCS point cloud internal structure

  auto fillPointSet = [] (const PointCloudSource& m, std::vector<gr::Point3D>& out) {
      out.clear();
      out.reserve(m.size());

      // TODO: copy other point-wise information, if any
      for(size_t i = 0; i< m.size(); i++){
          const auto& point = m[i];
          out.emplace_back(point.x, point.y, point.z);
          Eigen::Vector3f rgb(point.r, point.g, point.b);
          out[i].set_rgb(rgb);

          Eigen::Vector3f normal(point.normal_x, point.normal_y, point.normal_z);
          out[i].set_normal(normal);
          out[i].setProb(point.confidence);

      }
  };
  fillPointSet(*target_, set1);   // we extract base on P (scene)
  fillPointSet(*input_, set2);

  std::cout<<"super4pcs P size: "<<set1.size()<<std::endl;
  std::cout<<"super4pcs Q size: "<<set2.size()<<std::endl;

    //transformation set2(Q) to set1(P)
  float fitness_score_ = matcher.ComputeTransformation(set1, set2, final_transformation_, sampler, visitor);
  _pose_hypo = matcher._pose_hypo;
  _pose_lcp_scores = matcher._pose_lcp_scores;
  converged_ = true;
}
}

#endif

