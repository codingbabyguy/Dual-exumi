#ifndef _RM_DEFINE_H
#define _RM_DEFINE_H


#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cJSON.h"

#ifdef _WIN32
#define MSG_DONTWAIT 0
#include <winsock2.h>
#include <windows.h>

typedef SOCKET  SOCKHANDLE;
#endif

#ifdef __linux
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

typedef int SOCKHANDLE;
#endif

#ifdef __cplusplus
extern "C"
{
#endif
//////////////////////////////////////////////////////////////////////////////////
//睿尔曼智能科技有限公司        Author:Dong Qinpeng
//创建日期:2021/2/7
//版本：V1.1
//版权所有，盗版必究。
//Copyright(C) 睿尔曼智能科技有限公司
//All rights reserved
//文档说明：该文档定义了机械臂接口函数中使用到的结构体和错误代码类型
//////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C"
{
#endif

//机械臂自由度
#define   ARM_DOF               7              //机械臂自由度

//系统初始化错误代码
#define   SYS_OK                0x0000         //系统运行正常
#define JOINT_LIMIT_ERR         0x0001         //目标角度超过关节限位
#define INVERSE_KM_ERR          0x0002         //运动学逆解错误
#define M4_CTRL_ERR             0x0004         //M4内核通信错误
#define TOOL_BOARD_LOSS         0x0008         //末端工具接口板无法通信
#define INVERSE_KM_OVER_SPEED   0x0010         //逆解奇异点附近关节超速
#define ARM_COLLISION_ERR       0x0020         //机械臂发生碰撞
#define ARM_STOP_ERR            0x0040         //发生急停错误
#define QUEUE_INIT_ERR          0x0080         //轨迹规划点队列无法创建
#define SD_CARD_ERR             0x0100         //SD卡初始化错误
#define WIFI_INIT_ERR           0x0200         //WIFI模块初始化错误
#define CTRL_SYS_LOSS_ERR       0x0400         //实时层系统未按时上传数据
#define TEMPERATURE_INIT_ERR    0x0800         //温度传感器初始化错误
#define SYS_TEMPER_ERR          0x1000         //控制器过温
#define SYS_CURRENT_OVELOAD     0x2000         //控制器过流
#define SYS_VOLTAGE_OVELOAD     0x4000         //控制器过压
#define SYS_VOLTAGE_UNDER       0x8000         //控制器欠压


//机械臂关节错误类型
#define SYS_STATE_OK           0x0000         //系统正常
#define FOC_ERR                0x0001         //FOC频率过高
#define VOLTAGE_OVERLOAD       0x0002         //系统电压超过安全范围
#define VOLTAGE_UNDER          0x0004         //系统电压低于安全范围
#define TEMPER_OVERLOAD        0x0008         //温度过高
#define START_ERR              0x0010         //启动失败
#define ENCODER_ERR            0x0020         //编码器错误
#define CURRENT_OVERLOAD       0x0040         //电机电流超过安全范围
#define SOFTWARE_ERR           0x0080         //软件错误
#define TEMPER_DETECT_ERR      0x0100         //温度传感器错误
#define POS_LIMIT_ERR          0x0200         //位置超限错误
#define ERR_MASK_DRV8320       0x0400         //DRV8320错误
#define ERR_OVERLOAD           0x0800         //位置误差跟踪超限保护
#define CURRENT_DETECT_ERR     0x1000         //上电时电流传感器检测错误
#define BRAKE_ERR              0x2000         //抱闸错误
#define JOINT_CAN_LOSE_ERR     0xF000         //数据丢帧

typedef unsigned char byte;

//位姿结构体
typedef struct
{
    //位置
    float px;
    float py;
    float pz;
    //欧拉角
    float rx;
    float ry;
    float rz;
}POSE;

//坐标系
typedef struct
{
    char name[10];    //坐标系名称,不超过10个字符
}FRAME_NAME;

//坐标系
typedef struct
{
    FRAME_NAME frame_name;    //坐标系名称
    POSE pose;              //坐标系位姿
    float payload;     //坐标系末端负载重量
    float x;           //坐标系末端负载位置
    float y;           //坐标系末端负载位置
    float z;           //坐标系末端负载位置
}FRAME;

//机械臂控制模式
typedef enum
{
    None_Mode = 0,     //无规划
    Joint_Mode = 1,    //关节空间规划
    Line_Mode = 2,     //笛卡尔空间直线规划
    Circle_Mode = 3,   //笛卡尔空间圆弧规划
}ARM_CTRL_MODES;

//机械臂位置示教模式
typedef enum
{
    X_Dir = 0,       //X轴方向
    Y_Dir = 1,       //Y轴方向
    Z_Dir = 2,       //Z轴方向
}POS_TEACH_MODES;

//机械臂姿态示教模式
typedef enum
{
    RX_Rotate = 0,       //RX轴方向
    RY_Rotate = 1,       //RY轴方向
    RZ_Rotate = 2,       //RZ轴方向
}ORT_TEACH_MODES;

//控制器通讯方式选择
typedef enum
{
    WIFI_AP = 0,       //WIFI AP模式
    WIFI_STA = 1,      //WIFI STA模式
    BlueTeeth = 2,     //蓝牙模式
    USB       = 3,     //通过控制器UART-USB接口通信
    Ethernet  =4       //以太网口
}ARM_COMM_TYPE;

//机械臂状态参数
typedef struct
{
    float joint[ARM_DOF];         //关节角度
    float temperature[ARM_DOF];   //关节温度
    float voltage[ARM_DOF];       //关节电压
    float current[ARM_DOF];       //关节电流
    byte en_state[ARM_DOF];       //使能状态
    uint16_t err_flag[ARM_DOF];   //关节错误代码
    uint16_t sys_err;       //机械臂系统错误代码
}JOINT_STATE;

//位置
typedef struct
{
    //position
    float x;
    float y;
    float z;
    //orientation
    float w;
    float x_;
    float y_;
    float z_;
}POSE2;
//姿态
typedef struct
{
    float rx;
    float ry;
    float rz;
}ORT;
typedef struct
{
    POSE2 pose;
    ORT ort;
}KINEMATIC;
//旋转矩阵
typedef struct
{
    int irow;
    int iline;
    float data[4][4];
}Matrix;

#define  M_PI_RAD    0.0174533f
#define  MI_PI_ANG   57.2957805f
#define  PI          3.14159f

#define  M_PI		 3.14159265358979323846
#define  DELTA       0.26f   //关节判断角度差
#define  DELTA2      2*PI    //关节运动到该处



#ifdef __cplusplus
}
#endif
#endif
