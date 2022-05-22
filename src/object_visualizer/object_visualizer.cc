#include "object_visualizer/object_visualizer.h"
namespace kitti_visualizer {

ObjectVisualizer::ObjectVisualizer(ros::NodeHandle nh, ros::NodeHandle pnh)
    : nh_(nh), pnh_(pnh) {
  pnh_.param<std::string>("data_path", data_path_, "");
  pnh_.param<std::string>("dataset", dataset_, "");
  pnh_.param<int>("frame_size", frame_size_, 0);
  pnh_.param<int>("current_frame", current_frame_, 0);
  pnh_.param<bool>("save_image", save_image_, false);

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
  pub_bounding_boxes_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>(
      "kitti_visualizer/object/bounding_boxes", 2);

  if(save_image_)
  SaveVisualizerImage();
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
  for (const auto detection : detections) {
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
        tf::createQuaternionFromRPY(0.0 - detection[13], 0.0, 0.0 );
    tf::quaternionTFToMsg(bounding_box_quat, bounding_box.pose.orientation);
    // Bounding box dimensions
    bounding_box.dimensions.x = detection[7];
    bounding_box.dimensions.y = detection[9];
    bounding_box.dimensions.z = detection[8];
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
    if (object_type != "Car" && object_type != "Pedestrian" && object_type != "Vehicles" && object_type != "Bimo") continue;

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
    }else if(object_type == "Walkers") {
        detection.push_back(4.0);
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
