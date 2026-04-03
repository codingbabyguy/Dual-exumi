//
// Created by ubuntu on 22-6-22.
//
#include <ros/ros.h>
#include <rm_msgs/Gripper_Set.h>
#include <rm_msgs/Gripper_Pick.h>




int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_jiazhao");
    ros::NodeHandle nh;

    // 声明spinner对象，参数2表示并发线程数，默认处理全局Callback队列
    ros::AsyncSpinner spin(3);
    spin.start();

    ros::Duration(1.0).sleep();
    /*
     * 1.相关初始化
     */
    ros::Publisher test_Gripper_Set = nh.advertise<rm_msgs::Gripper_Set>("/rm_driver/Gripper_Set", 10);
    ros::Publisher test_Gripper_Pick = nh.advertise<rm_msgs::Gripper_Pick>("/rm_driver/Gripper_Pick", 10);

    ros::Duration(2.0).sleep();


    rm_msgs::Gripper_Pick Gripper_Pick;
    rm_msgs::Gripper_Set Gripper_Set;


    Gripper_Pick.speed = 500;
    Gripper_Pick.force = 200;

    Gripper_Set.position = 500;


    test_Gripper_Set.publish(Gripper_Set);
    ROS_INFO("Gripper_Set success!");
    ros::Duration(2.0).sleep();

    test_Gripper_Pick.publish(Gripper_Pick);
    ros::Duration(5.0).sleep();
    ROS_INFO("Gripper_Pick success!");

    test_Gripper_Set.publish(Gripper_Set);
    ROS_INFO("Gripper_Set success!");
    ros::Duration(2.0).sleep();

    

    ros::waitForShutdown();

    return 0;
}
