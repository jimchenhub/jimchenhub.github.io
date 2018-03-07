---
layout: post
title:  "ROS学习Navigation（二）"
date:   2017-08-11 19:30:00 +0800
categories: ROS, Navigation
---

紧接着上一篇继续实现第二部分。


## Setup and Configuration of the Navigation Stack on a Robot

在学习了基础的TF知识后，可以开始逐步学习如何publish tf的数据，Odometry的数据，sensor的数据，以及基础的navigation的设置。官方网站上写明了一个基础的navigation的tuning方法的[教程](http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide)，其中的cost map和Local Planer的部分是讲得稍微深入具体一些的，还没有完全看懂，但是相信日后会比较有参考价值。

首先来看一个整体的设计图，这个图非常的高层抽象，是很好的理解材料



![robot setup](http://wiki.ros.org/navigation/Tutorials/RobotSetup?action=AttachFile&do=get&target=overview_tf_small.png)

白色的框是已经被实现的必须的部分，灰色的已经被实现的可选部分，蓝色的是需要对不同机器人设计的部分。

### 1. Setup

1. 首先需要完成TF transform information 信息的发布，这个在前一大节已经讲解。
2. 机器人在世界中的导航是需要获取传感器信息来避障的，所以ROS假设这些信息是通过`sensor_msgs/LaserScan` 或 `sensor_msgs/PointCloud`来接收的。[Publishing Sensor Streams Over ROS](http://wiki.ros.org/navigation/Tutorials/RobotSetup/Sensors)
3. Odometry信息需要通过`tf`和 `nav_msgs/Odometry`来发布。[这里](http://wiki.ros.org/navigation/Tutorials/RobotSetup/Odom)是有相关的教程的。
4. Base Controller (base controller)。速度的指令是通过`geometry_msgs/Twist` message来传递给`cmd_vel` topic的。使用的frame是`base coordinate frame`。
5. Mapping (map_server) navigation本身是不需要map的，但是可以在 [building a map](http://wiki.ros.org/slam_gmapping/Tutorials/MappingFromLoggedData) 来创建一个运行环境的map。

### 2. Navigation Stack Setup

至此，已经通过tf publish了坐标系frame，可以接收到`sensor_msgs/LaserScan` 或 `sensor_msgs/PointCloud`的sensor信息，使用了tf和`nav_msgs/Odometry`发布了Odometry信息，同时接收了速度的命令。如果有任何一条未能满足，需要先完成上述的Setup部分。

#### 2.1 创建package
这个package要依赖于前面的所有package以及包含navigation stack高层接口 move_base package。

```
catkin_create_pkg my_robot_name_2dnav roscpp move_base my_tf_configuration_dep my_odom_configuration_dep my_sensor_configuration_dep
```

`my_tf_configuration_dep` 包括`tf`和` geometry_msgs`
`my_odom_configuration_dep ` 包括