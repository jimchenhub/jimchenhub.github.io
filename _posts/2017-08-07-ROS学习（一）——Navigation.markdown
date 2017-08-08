---
layout: post
title:  "ROS学习（一）——Navigation"
date:   2017-08-07 16:30:00 +0800
categories: Caffe, DAN
---

这次不想从头开始写ROS学习的部分，基础的很多概念都是学习的[这本书](http://wiki.ros.org/Books/Programming_Robots_with_ROS)，里面前面有很清楚的例子，也是围绕indigo这个版本来教学的，资料也非常多，做了一些简单的小bot。例如wander_bot, follow_bot等等。

但是对于坐标变换、图像处理、路径规划、地图导航等几个之后常用的方面还是了解甚少，所以特意把这几块的学习部分记录下来。

## Setting up robot using tf
这里主要参考ros wiki上的 [这篇文章](http://wiki.ros.org/navigation/Tutorials/RobotSetup/TF)。

### Basic Concepts
`TF`主要用于有很多不同的frame，而各自处理坐标变换极为复杂的情况。对于如下的简单例子，

![](http://wiki.ros.org/navigation/Tutorials/RobotSetup/TF?action=AttachFile&do=get&target=simple_robot.png)

有两个frame，`base_laser` 和 `base_link`，`TF`用一个树结构来保证两个frame间的单向连接。

![](http://wiki.ros.org/navigation/Tutorials/RobotSetup/TF?action=AttachFile&do=get&target=tf_robot.png)

首先要确定何为父节点何为儿子节点，再确定连接的方向。

### Writing codes
以`base_link`为父节点，我们将`base_laser`的数据转换到`base_link`上。首先创建一个package，使用 `roscpp`, `tf` 和 `geometry_msgs`.

```
$ cd %TOP_DIR_YOUR_CATKIN_WS%/src
$ catkin_create_pkg robot_setup_tf roscpp tf geometry_msgs
```

### Broadcasting a Transform
接下来要创建一个node把 `base_laser → base_link` 的变换广播给ROS。

```
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "robot_tf_publisher");
  ros::NodeHandle n;

  ros::Rate r(100);

  tf::TransformBroadcaster broadcaster;

  while(n.ok()){
    broadcaster.sendTransform(
      tf::StampedTransform(
        tf::Transform(tf::Quaternion(0, 0, 0, 1), tf::Vector3(0.1, 0.0, 0.2)),
        ros::Time::now(),"base_link", "base_laser"));
    r.sleep();
  }
}
```
The tf package provides an implementation of a **tf::TransformBroadcaster** to help make the task of publishing transforms easier. To use the TransformBroadcaster, we need to include the tf/transform_broadcaster.h header file.

Sending a transform with a **TransformBroadcaster requires five arguments**. First, we pass in the rotation transform, which is specified by a btQuaternion for any rotation that needs to occur between the two coordinate frames. In this case, we want to apply no rotation, so we send in a btQuaternion constructed from pitch, roll, and yaw values equal to zero. Second, a btVector3 for any translation that we'd like to apply. We do, however, want to apply a translation, so we create a btVector3 corresponding to the laser's x offset of 10cm and z offset of 20cm from the robot base. Third, we need to give the transform being published a timestamp, we'll just stamp it with ros::Time::now(). Fourth, we need to pass the name of the parent node of the link we're creating, in this case "base\_link." Fifth, we need to pass the name of the child node of the link we're creating, in this case "base\_laser."

### Using a Transform

```
#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <tf/transform_listener.h>

void transformPoint(const tf::TransformListener& listener){
  //we'll create a point in the base_laser frame that we'd like to transform to the base_link frame
  geometry_msgs::PointStamped laser_point;
  laser_point.header.frame_id = "base_laser";

  //we'll just use the most recent transform available for our simple example
  laser_point.header.stamp = ros::Time();

  //just an arbitrary point in space
  laser_point.point.x = 1.0;
  laser_point.point.y = 0.2;
  laser_point.point.z = 0.0;

  try{
    geometry_msgs::PointStamped base_point;
    listener.transformPoint("base_link", laser_point, base_point);

    ROS_INFO("base_laser: (%.2f, %.2f. %.2f) -----> base_link: (%.2f, %.2f, %.2f) at time %.2f",
        laser_point.point.x, laser_point.point.y, laser_point.point.z,
        base_point.point.x, base_point.point.y, base_point.point.z, base_point.header.stamp.toSec());
  }
  catch(tf::TransformException& ex){
    ROS_ERROR("Received an exception trying to transform a point from \"base_laser\" to \"base_link\": %s", ex.what());
  }
}

int main(int argc, char** argv){
  ros::init(argc, argv, "robot_tf_listener");
  ros::NodeHandle n;

  tf::TransformListener listener(ros::Duration(10));

  //we'll transform a point once every second
  ros::Timer timer = n.createTimer(ros::Duration(1.0), boost::bind(&transformPoint, boost::ref(listener)));

  ros::spin();

}
```

We'll create a function that, given a TransformListener, takes a point in the "base\_laser" frame and transforms it to the "base\_link" frame. This function will serve as a callback for the ros::Timer created in the main() of our program and will fire every second.

Here, we'll create our point as a `geometry_msgs::PointStamped`. The "Stamped" on the end of the message name just means that it includes a header, allowing us to associate both a timestamp and a frame\_id with the message. We'll set the stamp field of the laser\_point message to be ros::Time() which is a special time value that allows us to ask the TransformListener for the latest available transform. As for the frame\_id field of the header, we'll set that to be "base\_laser" because we're creating a point in the "base\_laser" frame.

Now that we have the point in the "base\_laser" frame we want to transform it into the "base\_link" frame. To do this, we'll use the TransformListener object, and call transformPoint() with three arguments: the name of the frame we want to transform the point to ("base\_link" in our case), the point we're transforming, and storage for the transformed point. So, after the call to transformPoint(), base\_point holds the same information as laser\_point did before only now in the "base\_link" frame.

### Build code
Now that we've written our little example, we need to build it. Open up the `CMakeLists.txt` file that is autogenerated by roscreate-pkg and add the following lines to the bottom of the file.

```
add_executable(tf_broadcaster src/tf_broadcaster.cpp)
add_executable(tf_listener src/tf_listener.cpp)
target_link_libraries(tf_broadcaster ${catkin_LIBRARIES})
target_link_libraries(tf_listener ${catkin_LIBRARIES})
```
