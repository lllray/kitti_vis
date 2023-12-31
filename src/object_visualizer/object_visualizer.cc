#include "object_visualizer/object_visualizer.h"
#include <iostream>
#include <stdlib.h>
namespace kitti_visualizer {

ObjectVisualizer::ObjectVisualizer(ros::NodeHandle nh, ros::NodeHandle pnh)
    : nh_(nh), pnh_(pnh) {
  pnh_.param<std::string>("data_path", data_path_, "");
  pnh_.param<std::string>("dataset", dataset_, "");
  pnh_.param<int>("frame_size", frame_size_, 0);
  pnh_.param<int>("current_frame", current_frame_, 0);
  pnh_.param<bool>("show_depth_cloud", show_depth_cloud_, false);
  pnh_.param<bool>("save_image", save_image_, false);
  pnh_.param<bool>("save_stereo_cloud", save_stereo_cloud_, false);

  // Judge whether the files number are valid
  AssertFilesNumber();

  // Subscriber
  sub_command_button_ =
      nh_.subscribe("/kitti_visualizer/command_button", 2,
                    &ObjectVisualizer::CommandButtonCallback, this);

  // Publisher
  pub_point_cloud_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZI>>(
      "kitti_visualizer/object/point_cloud", 2);
  pub_image_ =
      nh_.advertise<sensor_msgs::Image>("kitti_visualizer/object/image", 2);
  pub_depth_cloud_ =
      nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("kitti_visualizer/object/depth_cloud", 2);

//  pub_depth_cloud_ =
//          nh_.advertise<sensor_msgs::Image>("kitti_visualizer/object/depth", 2);

  pub_bounding_boxes_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>(
      "kitti_visualizer/object/bounding_boxes", 2);

  if(save_image_)
  SaveVisualizerImage();

  if(save_stereo_cloud_)
  StereoCloudSave();

//  PointCloudSave();
//  labelSave();
    ChangeImageSets();

}

void ObjectVisualizer::Visualizer() {
  // Get current file name
  std::ostringstream file_prefix;
  file_prefix << std::setfill('0') << std::setw(6) << current_frame_;
  ROS_INFO("Visualizing frame %s ...", file_prefix.str().c_str());

  // Visualize point cloud
  PointCloudVisualizer(file_prefix.str(), pub_point_cloud_);

  // Visualize image
  ImageVisualizer(file_prefix.str(), pub_image_);

  if(show_depth_cloud_)
  DepthImageVisualizer(file_prefix.str(), pub_depth_cloud_);

  // Visualize 3D bounding boxes
  BoundingBoxesVisualizer(file_prefix.str(), pub_bounding_boxes_);
}

void ObjectVisualizer::SaveVisualizerImage() {
    std::string image_file_name;
    std::string image_save_name;
    cv::Mat raw_image;
    // Get current file name
    std::string iamge_path = data_path_ + dataset_ + "/visualize_image/";
    std::string command = "mkdir -p " + iamge_path;
    std::system(command.c_str());

    for(int i=0;i<frame_size_;i++) {
        std::ostringstream file_prefix;
        file_prefix << std::setfill('0') << std::setw(6) << i;
        ROS_INFO("Visualizing frame %s ...", file_prefix.str().c_str());

        // Read image
        image_file_name =
                data_path_ + dataset_ + "/image_2/" + file_prefix.str() + ".png";
        image_save_name =
                data_path_ + dataset_ + "/visualize_image/" + file_prefix.str() + ".png";
        raw_image = cv::imread(image_file_name.c_str());

        // Draw 2D bounding boxes in image
        Draw2DBoundingBoxes(file_prefix.str(), raw_image);

        //Save image
        cv::imwrite(image_save_name, raw_image);
        ROS_INFO("Save Image %s ...", file_prefix.str().c_str());
    }
}
void ObjectVisualizer::PointCloudSave(){
        // Read point cloud
   for (int i=0;i<frame_size_;i++) {
       float height_offset = 0;
       if(i<2500){
           height_offset = i/500 * 5 + 10;
       }else{
           height_offset = (i - 2500) % 1500 / 300 * 5 + 10;
       }

       std::ostringstream file_prefix;
       file_prefix << std::setfill('0') << std::setw(6) << i;
       ROS_INFO("save cloud frame %s ...hight:%f", file_prefix.str().c_str(),height_offset);
       std::string cloud_file_name =
               data_path_ + dataset_ + "/velodyne/" + file_prefix.str() + ".bin";
       pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud(
               new pcl::PointCloud <pcl::PointXYZI>);
       ReadPointCloud(cloud_file_name, raw_cloud);
       std::string calib_file_name =
               data_path_ + dataset_ + "/calib/" + file_prefix.str() + ".txt";
       Eigen::MatrixXd trans_velo_to_cam = Eigen::MatrixXd::Identity(4, 4);
       ReadCalibMatrix(calib_file_name, "Tr_velo_to_cam:", trans_velo_to_cam);
       Eigen::MatrixXd trans_cam_to_rect = Eigen::MatrixXd::Identity(4, 4);
       ReadCalibMatrix(calib_file_name, "R0_rect:", trans_cam_to_rect);
       Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
       transform <<
                 0.f, 0.f, -1.f, 0.f,
               0.f, 1.f, 0.f, 0.f,
               -1.f, 0.f, 0.f, 0.f,
               0.f, 0.f, 0.f, 1.f;

       pcl::transformPointCloud(*raw_cloud, *raw_cloud, transform.inverse());

       std::string cloud_file_out =
               data_path_ + dataset_ + "/velodyne_transform/" + file_prefix.str() + ".bin";
       //Create & write .bin file
       std::ofstream out(cloud_file_out.c_str(), std::ios::out | std::ios::binary | std::ios::app);
       if (!out.good()) {
           std::cout << "Couldn't open " << cloud_file_out << std::endl;
           return;
       }

       for (size_t i = 0; i < raw_cloud->points.size(); ++i) {
           raw_cloud->points[i].z += height_offset;
           out.write((char *) &raw_cloud->points[i].x, 3 * sizeof(float));
           out.write((char *) &raw_cloud->points[i].intensity, sizeof(float));
       }
       out.close();
   }
};

void ObjectVisualizer::labelSave() {
    std::string command;
    command = "mkdir -p " + data_path_ + dataset_ + "/label_transform/";
    system(command.c_str());
    command = "mkdir -p " + data_path_ + dataset_ + "/calib_transform/";
    system(command.c_str());

    for (int i=0;i<frame_size_ ;i++) {
        std::ostringstream file_prefix;
        file_prefix << std::setfill('0') << std::setw(6) << i;
        ROS_INFO("save lable frame %s", file_prefix.str().c_str());
        std::string calib_file_name =
                data_path_ + dataset_ + "/calib/" + file_prefix.str() + ".txt";
        Eigen::MatrixXd trans_velo_to_cam = Eigen::MatrixXd::Identity(4, 4);
        ReadCalibMatrix(calib_file_name, "Tr_velo_to_cam:", trans_velo_to_cam);
        Eigen::MatrixXd trans_cam_to_rect = Eigen::MatrixXd::Identity(4, 4);
        ReadCalibMatrix(calib_file_name, "R0_rect:", trans_cam_to_rect);
        Eigen::MatrixXd trans_bev_to_ground = Eigen::MatrixXd::Identity(4, 4);
        ReadCalibMatrix(calib_file_name, "TR_bev_to_ground:", trans_bev_to_ground);
        Eigen::Matrix4d  transform = trans_cam_to_rect*trans_velo_to_cam*trans_bev_to_ground.inverse() * trans_velo_to_cam.inverse() *
                                     trans_cam_to_rect.inverse();

        std::vector<std::vector<float>> detections = ParseDetections(file_prefix.str());
        if (detections.size() == 0) {
            std::string file_out =
                    data_path_ + dataset_ + "/label_transform/" + file_prefix.str() + ".txt";
            std::ofstream out(file_out.c_str(), std::ios::out | std::ios::app);
            if (!out.good()) {
                std::cout << "Couldn't open " << file_out << std::endl;
                return;
            }
            continue;
        }
        for (std::vector<float> detection : detections) {
            jsk_recognition_msgs::BoundingBox bounding_box;
            // Bounding box position
            std::string file_out =
                    data_path_ + dataset_ + "/label_transform/" + file_prefix.str() + ".txt";
            std::ofstream out(file_out.c_str(), std::ios::out | std::ios::app);
            if (!out.good()) {
                std::cout << "Couldn't open " << file_out << std::endl;
                return;
            }

            Eigen::Vector4d rect_position(detection[10], detection[11], detection[12],
                                          1.0);
            rect_position = transform * rect_position;
            std::string object_type;
            if(detection.back() == 1.0) {
                object_type = "Car";
            }else if(detection.back() == 2.0) {
                object_type = "Pedestrian";
            }else if(detection.back() == 3.0) {
                object_type = "Bimo";
            }else if(detection.back() == 4.0) {
                object_type = "Truck";
            }else if(detection.back() == 5.0) {
                object_type = "Minicar";
            }else{
                continue;
            }
            out << object_type<<" "<< detection[0] << " " << detection[1] << " "
                << detection[2] << " " << detection[3] << " " << detection[4] << " "
                << detection[5] << " " << detection[6] << " " << detection[7] << " "
                << detection[8] << " " << detection[9] << " " << rect_position[0] << " "
                << rect_position[1] << " " << rect_position[2] << " " << detection[13] << std::endl;
        }
        std::string calib_out_file =
                data_path_ + dataset_ + "/calib_transform/" + file_prefix.str() + ".txt";
        std::ofstream out_calib(calib_out_file.c_str(), std::ios::out | std::ios::app);
        if (!out_calib.good()) {
            std::cout << "Couldn't open " << calib_file_name << std::endl;
            return;
        }
        std::ifstream ifs(calib_file_name);
        if (!ifs) {
            ROS_ERROR("File %s does not exist", calib_file_name.c_str());
            ros::shutdown();
        }
        std::string temp_str;
        while (std::getline(ifs, temp_str)) {
            std::istringstream iss(temp_str);
            std::string name;
            iss >> name;
            if (name == "Tr_velo_to_cam:") {
                Eigen::Matrix4d  transform1 = trans_velo_to_cam*trans_bev_to_ground.inverse();
                out_calib << "Tr_velo_to_cam: "
                          << transform1(0,0)<<" "<< transform1(0,1)<<" "<< transform1(0,2)<<" "<< transform1(0,3)<<" "
                          << transform1(1,0)<<" "<< transform1(1,1)<<" "<< transform1(1,2)<<" "<< transform1(1,3)<<" "
                          << transform1(2,0)<<" "<< transform1(2,1)<<" "<< transform1(2,2)<<" "<< transform1(2,3)<< std::endl;
            }else{
                out_calib << temp_str << std::endl;
            }
        }
    }
}

void ObjectVisualizer::ChangeImageSets(){
    std::string command;
    command = "mkdir -p " + data_path_ + dataset_ + "/image_sets_20m/";
    system(command.c_str());
    for (int i=0;i<frame_size_;i++) {
        // Read transform matrixs from calib file
        std::ostringstream file_prefix;
        file_prefix << std::setfill('0') << std::setw(6) << i;

        std::string calib_file_name =
                data_path_ + dataset_ + "/calib/" + file_prefix.str() + ".txt";
        Eigen::MatrixXd trans_bev_to_ground = Eigen::MatrixXd::Identity(4, 4);
        ReadCalibMatrix(calib_file_name, "TR_bev_to_ground:", trans_bev_to_ground);
        float altitude = trans_bev_to_ground(2,3);
        ROS_INFO("save stereo cloud frame %s altitude:%f", file_prefix.str().c_str(),altitude);
        std::string train_file_out =
                data_path_ + dataset_ + "/image_sets_20m/train.txt";
        std::string val_file_out =
                data_path_ + dataset_ + "/image_sets_20m/val.txt";
        std::string trainval_file_out =
                data_path_ + dataset_ + "/image_sets_20m/trainval.txt";

        //Create & write .bin file
        std::ofstream train_out(train_file_out.c_str(), std::ios::out | std::ios::binary | std::ios::app);
        if (!train_out.good()) {
            std::cout << "Couldn't open " << train_file_out << std::endl;
            return;
        }
        std::ofstream val_out(val_file_out.c_str(), std::ios::out | std::ios::binary | std::ios::app);
        if (!val_out.good()) {
            std::cout << "Couldn't open " << val_file_out << std::endl;
            return;
        }
        std::ofstream trainval_out(trainval_file_out.c_str(), std::ios::out | std::ios::binary | std::ios::app);
        if (!trainval_out.good()) {
            std::cout << "Couldn't open " << trainval_file_out << std::endl;
            return;
        }
        if(altitude<=20){
            if(i%2==0)
                train_out << file_prefix.str()<<std::endl;
            else
                val_out << file_prefix.str()<<std::endl;
            trainval_out << file_prefix.str()<<std::endl;
        }
    }
}

void ObjectVisualizer::StereoCloudSave(){
    float intensity = 0.f;

    std::string command;
    command = "mkdir -p " + data_path_ + dataset_ + "/stereo_cloud/";
    system(command.c_str());
    for (int i=0;i<frame_size_;i++) {
        // Read transform matrixs from calib file
        std::ostringstream file_prefix;
        file_prefix << std::setfill('0') << std::setw(6) << i;
        ROS_INFO("save stereo cloud frame %s", file_prefix.str().c_str());
        std::string calib_file_name =
                data_path_ + dataset_ + "/calib/" + file_prefix.str() + ".txt";
        Eigen::MatrixXd trans_velo_to_cam = Eigen::MatrixXd::Identity(4, 4);
        ReadCalibMatrix(calib_file_name, "Tr_velo_to_cam:", trans_velo_to_cam);
        Eigen::MatrixXd trans_cam_to_rect = Eigen::MatrixXd::Identity(4, 4);
        ReadCalibMatrix(calib_file_name, "R0_rect:", trans_cam_to_rect);
        Eigen::MatrixXd trans_bev_to_ground = Eigen::MatrixXd::Identity(4, 4);
        ReadCalibMatrix(calib_file_name, "TR_bev_to_ground:", trans_bev_to_ground);
        Eigen::MatrixXd P2 = Eigen::MatrixXd::Identity(3, 4);
        ReadCalibMatrix(calib_file_name, "P2:", P2);
        const float focus = P2(0,0);
        const float cx = P2(0,2);
        const float cy = P2(1,2);
        const float fb = 0.1 * focus;


        Eigen::MatrixXd transform = trans_bev_to_ground * trans_velo_to_cam.inverse() *
                                    trans_cam_to_rect.inverse();

        // Read image
        // Read image
        std::string image_file_name =
                data_path_ + dataset_ + "/stereo_depth/" + file_prefix.str() + ".tiff";
        std::string left_file_name =
                data_path_ + dataset_ + "/image_2/" + file_prefix.str() + ".png";
        std::cout<<"load:"<<image_file_name<<std::endl;
        std::cout<<"load:"<<left_file_name<<std::endl;
        cv::Mat raw_image = cv::imread(image_file_name.c_str(),cv::IMREAD_UNCHANGED);
        cv::Mat left_image = cv::imread(left_file_name.c_str());
        if(raw_image.empty()){
            std::cout<<"depth image is empty!"<<std::endl;
            return;
        }
        if(left_image.empty()){
            std::cout<<"left image is empty!"<<std::endl;
            return;
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud(
                new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointXYZRGB target_pt;
        float dsp_val;
        cv::Vec3i bgr;
        for (int v = 0; v < raw_image.rows; v ++) {
            for (int u = 0; u < raw_image.cols; u ++) {
                dsp_val = raw_image.at<float>(v, u);
                bgr = left_image.at<cv::Vec3b>(v,u);
                target_pt.z = fb/dsp_val;
                if ( target_pt.z > 0) {
                    target_pt.x = (u - cx)*target_pt.z/focus;
                    target_pt.y = (v - cy)*target_pt.z/focus;
                    target_pt.b = bgr.val[0];
                    target_pt.g = bgr.val[1];
                    target_pt.r = bgr.val[2];
                    raw_cloud->push_back(target_pt);
                }
            }
        }
        pcl::transformPointCloud(*raw_cloud, *raw_cloud, transform.cast <float>());

        std::string cloud_file_out =
                data_path_ + dataset_ + "/stereo_cloud/" + file_prefix.str() + ".bin";
        //Create & write .bin file
        std::ofstream out(cloud_file_out.c_str(), std::ios::out | std::ios::binary | std::ios::app);
        if (!out.good()) {
            std::cout << "Couldn't open " << cloud_file_out << std::endl;
            return;
        }

        for (size_t i = 0; i < raw_cloud->points.size(); ++i) {
            out.write((char *) &raw_cloud->points[i].x, 3 * sizeof(float));
            out.write((char *) &intensity, sizeof(float));
            out.write((char *) &raw_cloud->points[i].rgb, sizeof(float));
        }
        out.close();
    }
}

void ObjectVisualizer::PointCloudVisualizer(const std::string& file_prefix,
                                            const ros::Publisher publisher) {
  // Read point cloud
  std::string cloud_file_name =
      data_path_ + dataset_ + "/velodyne/" + file_prefix + ".bin";
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  ReadPointCloud(cloud_file_name, raw_cloud);
  // Publish point cloud
  raw_cloud->header.frame_id = "base_link";
  publisher.publish(raw_cloud);
}

void ObjectVisualizer::ImageVisualizer(const std::string& file_prefix,
                                       const ros::Publisher publisher) {
  // Read image
  std::string image_file_name =
      data_path_ + dataset_ + "/image_2/" + file_prefix + ".png";
  cv::Mat raw_image = cv::imread(image_file_name.c_str());

  // Draw 2D bounding boxes in image
  Draw2DBoundingBoxes(file_prefix, raw_image);

  // Publish image
  sensor_msgs::ImagePtr raw_image_msg =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", raw_image).toImageMsg();
  raw_image_msg->header.frame_id = "base_link";
  publisher.publish(raw_image_msg);
}

void ObjectVisualizer::DepthImageVisualizer(const std::string& file_prefix,
                                           const ros::Publisher publisher) {
    // Read transform matrixs from calib file
    std::string calib_file_name =
            data_path_ + dataset_ + "/calib/" + file_prefix + ".txt";
    Eigen::MatrixXd trans_velo_to_cam = Eigen::MatrixXd::Identity(4, 4);
    ReadCalibMatrix(calib_file_name, "Tr_velo_to_cam:", trans_velo_to_cam);
    Eigen::MatrixXd trans_cam_to_rect = Eigen::MatrixXd::Identity(4, 4);
    ReadCalibMatrix(calib_file_name, "R0_rect:", trans_cam_to_rect);
    Eigen::MatrixXd trans_bev_to_ground = Eigen::MatrixXd::Identity(4, 4);
    ReadCalibMatrix(calib_file_name, "TR_bev_to_ground:", trans_bev_to_ground);
    Eigen::MatrixXd P2 = Eigen::MatrixXd::Identity(3, 4);
    ReadCalibMatrix(calib_file_name, "P2:", P2);
    const float focus = P2(0,0);
    const float cx = P2(0,2);
    const float cy = P2(1,2);
    const float fb = 0.1 * focus;


    Eigen::MatrixXd transform = trans_bev_to_ground * trans_velo_to_cam.inverse() *
                                    trans_cam_to_rect.inverse();

    // Read image
    std::string image_file_name =
            data_path_ + dataset_ + "/stereo_depth/" + file_prefix + ".tiff";
    std::string left_file_name =
            data_path_ + dataset_ + "/image_2/" + file_prefix + ".png";
    std::cout<<"load:"<<image_file_name<<std::endl;
    std::cout<<"load:"<<left_file_name<<std::endl;
    cv::Mat raw_image = cv::imread(image_file_name.c_str(),cv::IMREAD_UNCHANGED);
    cv::Mat left_image = cv::imread(left_file_name.c_str());
    if(raw_image.empty()){
        std::cout<<"depth image is empty!"<<std::endl;
        return;
    }
    if(left_image.empty()){
        std::cout<<"left image is empty!"<<std::endl;
        return;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZRGB target_pt;
    float dsp_val;
    cv::Vec3i bgr;
    for (int v = 0; v < raw_image.rows; v +=2) {
        for (int u = 0; u < raw_image.cols; u +=2) {
            dsp_val = raw_image.at<float>(v, u);
            bgr = left_image.at<cv::Vec3b>(v,u);
            target_pt.z = fb/dsp_val;
            if ( target_pt.z > 0) {
                target_pt.x = (u - cx)*target_pt.z/focus;
                target_pt.y = (v - cy)*target_pt.z/focus;
                target_pt.b = bgr.val[0];
                target_pt.g = bgr.val[1];
                target_pt.r = bgr.val[2];
                raw_cloud->push_back(target_pt);
            }
        }
    }
    pcl::transformPointCloud(*raw_cloud, *raw_cloud, transform.cast <float>());
    raw_cloud->header.frame_id = "base_link";
    publisher.publish(raw_cloud);
    // Publish image
//    sensor_msgs::ImagePtr raw_image_msg =
//            cv_bridge::CvImage(std_msgs::Header(), "mono16", raw_image).toImageMsg();
//    raw_image_msg->header.frame_id = "base_link";
//    publisher.publish(raw_image_msg);
}

void ObjectVisualizer::Draw2DBoundingBoxes(const std::string& file_prefix,
                                           cv::Mat& raw_image) {
  // Read bounding boxes data
  std::vector<std::vector<float>> detections = ParseDetections(file_prefix);

  // Draw bounding boxes in image
  for (const auto detection : detections) {
      cv::Scalar color;
      if(detection.back() == 1.0) {
          color = cv::Scalar(255, 0, 0);//"Car"
      }else if(detection.back() == 2.0) {
          color = cv::Scalar(0, 255, 0);//"Pedestrian"
      }else if(detection.back() == 3.0) {
          color = cv::Scalar(0, 0, 255);//"Bimo"
      }else if(detection.back() == 4.0) {
          color = cv::Scalar(0, 0, 0);//"None"
      }
    cv::rectangle(raw_image, cv::Point(detection[3], detection[4]),
                  cv::Point(detection[5], detection[6]), color,
                  2, 8, 0);

  }
}

void ObjectVisualizer::BoundingBoxesVisualizer(const std::string& file_prefix,
                                               const ros::Publisher publisher) {
  // Read bounding boxes data
  std::vector<std::vector<float>> detections = ParseDetections(file_prefix);

  // Transform bounding boxes to jsk_recognition_msgs
  jsk_recognition_msgs::BoundingBoxArray bounding_box_array =
      TransformBoundingBoxes(detections, file_prefix);

  // Publish bounding boxes
  bounding_box_array.header.frame_id = "base_link";
  pub_bounding_boxes_.publish(bounding_box_array);
}

jsk_recognition_msgs::BoundingBoxArray ObjectVisualizer::TransformBoundingBoxes(
    const std::vector<std::vector<float>> detections,
    const std::string& file_prefix) {
  // Read transform matrixs from calib file
  std::string calib_file_name =
      data_path_ + dataset_ + "/calib/" + file_prefix + ".txt";
  Eigen::MatrixXd trans_velo_to_cam = Eigen::MatrixXd::Identity(4, 4);
  ReadCalibMatrix(calib_file_name, "Tr_velo_to_cam:", trans_velo_to_cam);
  Eigen::MatrixXd trans_cam_to_rect = Eigen::MatrixXd::Identity(4, 4);
  ReadCalibMatrix(calib_file_name, "R0_rect:", trans_cam_to_rect);

  // Set bounding boxes to jsk_recognition_msgs::BoundingBoxArray
  jsk_recognition_msgs::BoundingBoxArray bounding_box_array;
  for (std::vector<float> detection : detections) {
    jsk_recognition_msgs::BoundingBox bounding_box;
    // Bounding box position
    Eigen::Vector4d rect_position(detection[10], detection[11], detection[12],
                                  1.0);
    Eigen::MatrixXd velo_position = trans_velo_to_cam.inverse() *
                                    trans_cam_to_rect.inverse() * rect_position;

    bounding_box.pose.position.x = velo_position(0);
    bounding_box.pose.position.y = velo_position(1);
    bounding_box.pose.position.z = velo_position(2) + detection[7] / 2.0;
    // Bounding box orientation
    tf::Quaternion bounding_box_quat =
        tf::createQuaternionFromRPY(0.0, 0.0, 0.0 - detection[13] );
    tf::quaternionTFToMsg(bounding_box_quat, bounding_box.pose.orientation);
    // Bounding box dimensions
    bounding_box.dimensions.x = detection[8];//kitti width 8 = livox height
    bounding_box.dimensions.y = detection[9];//kitti long  9 =
    bounding_box.dimensions.z = detection[7];//kitt height 7 = livox height
    // Bounding box header
    bounding_box.header.stamp = ros::Time::now();
    bounding_box.header.frame_id = "base_link";
    bounding_box_array.boxes.push_back(bounding_box);
  }

  return bounding_box_array;
}

std::vector<std::vector<float>> ObjectVisualizer::ParseDetections(
    const std::string& file_prefix) {
  // Open bounding boxes file
  std::string detections_file_name;
  if (dataset_ == "training") {
    detections_file_name =
        data_path_ + dataset_ + "/label_2/" + file_prefix + ".txt";
  } else if (dataset_ == "testing") {
    detections_file_name =
        data_path_ + dataset_ + "/results/" + file_prefix + ".txt";
  }
  std::ifstream detections_file(detections_file_name);
  if (!detections_file) {
    ROS_ERROR("File %s does not exist", detections_file_name.c_str());
    ros::shutdown();
  }

  // Parse objects data
  std::vector<std::vector<float>> detections;
  std::string line_str;
  while (getline(detections_file, line_str)) {
    // Store std::string into std::stringstream
    std::stringstream line_ss(line_str);
    // Parse object type
    std::string object_type;
    getline(line_ss, object_type, ' ');
    // Parse object data
    std::vector<float> detection;
    std::string str;
    while (getline(line_ss, str, ' ')) {
      detection.push_back(boost::lexical_cast<float>(str));
    }
    if(object_type == "Car") {
        detection.push_back(1.0);
    }else if(object_type == "Pedestrian") {
        detection.push_back(2.0);
    }else if(object_type == "Bimo") {
        detection.push_back(3.0);
    }else if(object_type == "Truck") {
        detection.push_back(4.0);
    }else if(object_type == "Minicar") {
        detection.push_back(5.0);
    }
    detections.push_back(detection);
  }

  return detections;
}

void ObjectVisualizer::CommandButtonCallback(
    const std_msgs::String::ConstPtr& in_command) {
  // Parse frame number form command
  if (in_command->data == "Next") {
    current_frame_ = (frame_size_ + current_frame_ + 1) % frame_size_;
  } else if (in_command->data == "Prev") {
    current_frame_ = (frame_size_ + current_frame_ - 1) % frame_size_;
  } else {
    int frame = std::stoi(in_command->data);
    if (frame >= 0 && frame < frame_size_)
      current_frame_ = frame;
    else
      ROS_ERROR("No frame %s", in_command->data.c_str());
  }

  // Visualize object data
  Visualizer();
}

void ObjectVisualizer::AssertFilesNumber() {
  if(frame_size_<=0)
  frame_size_ =  FolderFilesNumber(data_path_ + dataset_ + "/velodyne");
  // Assert velodyne files numbers
  ROS_ASSERT(FolderFilesNumber(data_path_ + dataset_ + "/velodyne") ==
             frame_size_);
  // Assert image_2 files numbers
  ROS_ASSERT(FolderFilesNumber(data_path_ + dataset_ + "/image_2") ==
             frame_size_);
  // Assert calib files numbers
  ROS_ASSERT(FolderFilesNumber(data_path_ + dataset_ + "/calib") ==
             frame_size_);
  if (dataset_ == "training") {
    // Assert label_2 files numbers
    ROS_ASSERT(FolderFilesNumber(data_path_ + dataset_ + "/label_2") ==
               frame_size_);
  } else if (dataset_ == "testing") {
    // Assert results files numbers
    ROS_ASSERT(FolderFilesNumber(data_path_ + dataset_ + "/results") ==
               frame_size_);
  } else {
    ROS_ERROR("Dataset input error: %s", dataset_.c_str());
    ros::shutdown();
  }
    ROS_INFO("%d data input success from: %s",frame_size_, dataset_.c_str());
}
}
