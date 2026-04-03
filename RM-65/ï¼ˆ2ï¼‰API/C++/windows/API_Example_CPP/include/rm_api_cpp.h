#ifndef RM_API_CPP_H
#define RM_API_CPP_H

#include "rm_api.h"
class RM_API_CPP
{

public:
    ///
    /// \brief RM_API   构造函数
    /// \param Dev_Model 设备型号:63/65/75
    ///  RM_APISHARED_EXPORT
    ///
    RM_APISHARED_EXPORT RM_API_CPP(int Dev_Model);

    ///
    /// \brief API_Version_Cpp 查询api版本信息
    /// \return     api版本号
    ///
    RM_APISHARED_EXPORT char * API_Version_Cpp();

    ///
    /// \brief Arm_Socket_Start_Cpp 连接机械臂
    /// \param arm_ip  ip地址
    /// \param arm_prot 端口号
    /// \param recv_timeout 设置接收机械臂数据的超时时间，防止程序一直阻塞，单位ms。建议20ms左右
    /// \return 成功:0: 失败 < 0
    ///
    RM_APISHARED_EXPORT int Arm_Socket_Start_Cpp(char* arm_ip,int arm_port,int recv_timeout);

    ///
    /// \brief Arm_Sockrt_State_Cpp  查询机械臂连接状态
    /// \return     0-连接   1-中断
    ///
    RM_APISHARED_EXPORT int Arm_Sockrt_State_Cpp();

    ///
    /// \brief Arm_Socket_Close_Cpp 关闭与机械臂的socket连接
    ///
    RM_APISHARED_EXPORT void Arm_Socket_Close_Cpp();

    ///
    /// \brief Arm_Socket_Buffer_Clear_Cpp 清理socket接收缓存内的数据
    ///
    RM_APISHARED_EXPORT void Arm_Socket_Buffer_Clear_Cpp();

    ///
    /// \brief Set_Joint_Speed_Cpp 设置关节最大速度
    /// \param joint_num 关节序号，1~7
    /// \param speed 关节转速，单位：RPM
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Joint_Speed_Cpp(byte joint_num, float speed, bool block);

    ///
    /// \brief Set_Joint_Acc_Cpp 设置关节最大加速度
    /// \param joint_num 关节序号，1~7
    /// \param acc 关节转速，单位：RPM/s
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Joint_Acc_Cpp(byte joint_num, float acc, bool block);

    ///
    /// \brief Set_Joint_Min_Pos_Cpp 设置关节最小限位
    /// \param joint_num 关节序号，1~7
    /// \param joint 关节最小位置，单位：°
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Joint_Min_Pos_Cpp(byte joint_num, float joint, bool block);

    ///
    /// \brief Set_Joint_Max_Pos_Cpp 设置关节最大限位
    /// \param joint_num 关节序号，1~7
    /// \param joint 关节最大位置，单位：°
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Joint_Max_Pos_Cpp(byte joint_num, float joint, bool block);

    ///
    /// \brief Set_Joint_EN_State_Cpp 设置关节使能状态
    /// \param joint_num 关节序号，1~7
    /// \param state true-上使能，false-掉使能
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Joint_EN_State_Cpp(byte joint_num, bool state, bool block);

    ///
    /// \brief Set_Joint_Zero_Pos_Cpp 将当前位置设置为关节零位
    /// \param joint_num 关节序号，1~7
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return
    ///
    RM_APISHARED_EXPORT int Set_Joint_Zero_Pos_Cpp(byte joint_num, bool block);

    ///
    /// \brief Set_Joint_Err_Clear_Cpp 清除关节错误代码
    /// \param joint_num 关节序号，1~7
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Joint_Err_Clear_Cpp(byte joint_num, bool block);

    ///
    /// \brief Start_Calibrate_Cpp   开始标定关节
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Start_Calibrate_Cpp(bool block);

    ///
    /// \brief Stop_Calibrate_Cpp   停止标定关节
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_Calibrate_Cpp(bool block);

    ///
    /// \brief Get_Calibrate_State_Cpp 获取标定状态
    /// \param joint_state  关节状态
    /// \param state    报错信息
    /// \param time  标定时间
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Calibrate_State_Cpp(float* joint_state,uint16_t *state,int *time,bool block);

    ///
    /// \brief Get_Joint_Speed_Cpp 查询关节最大速度
    /// \param speed 关节1~7转速数组，单位：RPM
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Speed_Cpp(float *speed);

    ///
    /// \brief Get_Joint_Acc_Cpp 查询关节最大加速度
    /// \param acc 关节1~7加速度数组，单位：RPM/s
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Acc_Cpp(float *acc);

    ///
    /// \brief Get_Joint_Min_Pos_Cpp 获取关节最小位置
    /// \param min_joint 关节1~7最小位置数组，单位：°
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Min_Pos_Cpp(float *min_joint);

    ///
    /// \brief Get_Joint_Max_Pos_Cpp 获取关节最大位置
    /// \param max_joint 关节1~7最大位置数组，单位：°
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Max_Pos_Cpp(float *max_joint);

    ///
    /// \brief Get_Joint_EN_State_Cpp 获取关节使能状态
    /// \param state 关节1~7使能状态数组，1-使能状态，0-掉使能状态
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_EN_State_Cpp(byte *state);

    ///
    /// \brief Get_Joint_Err_Flag_Cpp 获取关节Err Flag
    /// \param state 关节1~7使能状态数组，1-使能状态，0-掉使能状态
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Err_Flag_Cpp(uint16_t *state);

    ///
    /// \brief Get_Joint_Software_Version_Cpp       查询关节软件版本号
    /// \param version                          获取到的关节软件版本号
    /// \param block                            0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                                 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Software_Version_Cpp(float* version);

    ///
    /// \brief Set_Arm_Line_Speed_Cpp 设置机械臂末端最大线速度
    /// \param speed 末端最大线速度，单位m/s
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Line_Speed_Cpp(float speed, bool block);

    ///
    /// \brief Set_Arm_Line_Acc_Cpp 设置机械臂末端最大线加速度
    /// \param acc 末端最大线加速度，单位m/s^2
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Line_Acc_Cpp(float acc, bool block);

    ///
    /// \brief Set_Arm_Angular_Speed_Cpp 设置机械臂末端最大角速度
    /// \param speed 末端最大角速度，单位rad/s
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Angular_Speed_Cpp(float speed, bool block);

    ///
    /// \brief Set_Arm_Angular_Acc_Cpp 设置机械臂末端最大角加速度
    /// \param acc 末端最大角加速度，单位rad/s^2
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Angular_Acc_Cpp(float acc, bool block);

    ///
    /// \brief Get_Arm_Line_Speed_Cpp 获取机械臂末端最大线速度
    /// \param speed 末端最大线速度，单位m/s
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Line_Speed_Cpp(float *speed);

    ///
    /// \brief Get_Arm_Line_Acc_Cpp 获取机械臂末端最大线加速度
    /// \param acc 末端最大线加速度，单位m/s^2
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Line_Acc_Cpp(float *acc);

    ///
    /// \brief Get_Arm_Angular_Speed_Cpp 获取机械臂末端最大角速度
    /// \param speed 末端最大角速度，单位rad/s
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Angular_Speed_Cpp(float *speed);

    ///
    /// \brief Get_Arm_Angular_Acc_Cpp 获取机械臂末端最大角加速度
    /// \param acc 末端最大角加速度，单位rad/s^2
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Angular_Acc_Cpp(float *acc);

    ///
    /// \brief Set_Arm_Tip_Init_Cpp 设置机械臂末端参数为初始值
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Tip_Init_Cpp(bool block);

    ///
    /// \brief Set_Collision_Stage_Cpp 设置机械臂动力学碰撞检测等级
    /// \param stage 等级：0~8，0-无碰撞，8-碰撞最灵敏
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Collision_Stage_Cpp(int stage, bool block);

    ///
    /// \brief Get_Collision_Stage_Cpp 查询碰撞防护等级
    /// \param stage    等级，范围：0~8
    /// \param block    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Collision_Stage_Cpp(int *stage);

    ///
    /// \brief Set_DH_Data_Cpp 该函数用于重新设置DH参数，一般在标定后使用
    /// \param lsb 相关DH参数，单位：mm
    /// \param lse 相关DH参数，单位：mm
    /// \param lew 相关DH参数，单位：mm
    /// \param lwt 相关DH参数，单位：mm
    /// \param d3  相关DH参数，单位：mm
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_DH_Data_Cpp(float lsb, float lse, float lew, float lwt, float d3, bool block);

    ///
    /// \brief Get_DH_Data_Cpp 该函数用于获取DH参数，一般在标定后使用
    /// \param lsb 获取到的DH参数，单位：mm
    /// \param lse 获取到的DH参数，单位：mm
    /// \param lew 获取到的DH参数，单位：mm
    /// \param lwt 获取到的DH参数，单位：mm
    /// \param d3  获取到的DH参数，单位：mm
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_DH_Data_Cpp(float* lsb, float* lse, float* lew, float* lwt, float* d3);

    ///
    /// \brief Set_Joint_Zero_Offset_Cpp 该函数用于设置机械臂各关节零位补偿角度，一般在对机械臂零位进行标定后调用该函数
    /// \param offset 关节1~6的零位补偿角度数组, 单位：度
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Joint_Zero_Offset_Cpp(const float *offset, bool block);

    ///
    /// \brief          Set_Arm_Dynamic_Parm_Cpp 重新设置机械臂动力学参数
    /// \param parm     设置机械臂动力学参数，
    /// \param block    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0       -成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Dynamic_Parm_Cpp(const float* parm, bool block);

    ///
    /// \brief Get_Tool_Software_Version_Cpp        查询末端接口板软件版本号
    /// \param version                          查询到的末端接口板软件版本号
    /// \param block                            0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                                 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Tool_Software_Version_Cpp(int* version);

    ///
    /// \brief Set_Arm_Servo_State_Cpp 打开或者关闭控制器对机械臂伺服状态查询
    /// \param cmd true-打开，  false-关闭
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Servo_State_Cpp(bool cmd, bool block);

    ///
    /// \brief Set_System_State_Servo_Cpp 该函数用于打开或者关闭系统状态自动回传，自动回传打开后，以50Hz的频率周期回传系统状态。
    /// \param cmd true-开启自动回传，false
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_System_State_Servo_Cpp(bool cmd, bool block);

    ///
    /// \brief Auto_Set_Tool_Frame_Cpp 六点法自动设置工具坐标系 标记点位
    /// \param point_num 1~6代表6个标定点
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 说明：控制器只能存储十个工具，超过十个控制器不予响应，请在标定前查询已有工具
    RM_APISHARED_EXPORT int Auto_Set_Tool_Frame_Cpp(byte point_num, bool block);

    ///
    /// \brief Generate_Auto_Tool_Frame_Cpp 六点法自动设置工具坐标系 提交
    /// \param name  工具坐标系名称，不能超过十个字节。
    /// \param payload 新工具执行末端负载重量  单位g
    /// \param x 新工具执行末端负载重量 位置x
    /// \param y 新工具执行末端负载重量 位置y
    /// \param z 新工具执行末端负载重量 位置z
    /// \param block block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Generate_Auto_Tool_Frame_Cpp(const char *name,float payload,float x,float y,float z, bool block);

    ///
    /// \brief Manual_Set_Tool_Frame_Cpp 手动设置工具坐标系
    /// \param name 工具坐标系名称，不能超过十个字节。
    /// \param pose 新工具执行末端相对于机械臂法兰中心的位姿
    /// \param payload 新工具执行末端负载重量  单位g
    /// \param x 新工具执行末端负载重量 位置x
    /// \param y 新工具执行末端负载重量 位置y
    /// \param z 新工具执行末端负载重量 位置z
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 控制器只能存储十个工具，超过十个控制器不予响应，请在标定前查询已有工具
    RM_APISHARED_EXPORT int Manual_Set_Tool_Frame_Cpp(const char *name, POSE pose,float payload,float x,float y,float z, bool block);

    ///
    /// \brief Change_Tool_Frame_Cpp 切换当前工具坐标系
    /// \param name 目标工具坐标系名称
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Change_Tool_Frame_Cpp(const char *name, bool block);

    ///
    /// \brief Delete_Tool_Frame_Cpp 删除指定工具坐标系
    /// \param name 要删除的工具坐标系名称
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 备注：删除坐标系后，机械臂将切换到机械臂法兰末端工具坐标系
    RM_APISHARED_EXPORT int Delete_Tool_Frame_Cpp(const char *name, bool block);

    ///
    /// \brief Set_Payload_Cpp 设置末端负载质量和质心
    /// \param payload 负载质量，单位：g，最高不超过5000g
    /// \param cx 末端负载质心位置，单位：mm
    /// \param cy 末端负载质心位置，单位：mm
    /// \param cz 末端负载质心位置，单位：mm
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Payload_Cpp(int payload, float cx, float cy, float cz, bool block);

    ///
    /// \brief Set_None_Payload_Cpp 设置末端无负载
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_None_Payload_Cpp(bool block);

    ///
    /// \brief Get_Current_Tool_Frame_Cpp 获取当前工具坐标系
    /// \param tool 返回的坐标系
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Current_Tool_Frame_Cpp(FRAME *tool);

    ///
    /// \brief Get_Given_Tool_Frame_Cpp 获取指定工具坐标系
    /// \param name 指定的工具名称
    /// \param tool 返回的工具参数
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Given_Tool_Frame_Cpp(const char *name, FRAME *tool);

    ///
    /// \brief Get_All_Tool_Frame_Cpp 获取所有工具坐标系名称
    /// \param names 返回的工具名称数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_All_Tool_Frame_Cpp(FRAME_NAME *names);

    ///
    /// \brief Auto_Set_Work_Frame_Cpp 三点法自动设置工作坐标系
    /// \param name 工作坐标系名称，不能超过十个字节。
    /// \param point_num 1~3代表3个标定点，依次为原点、X轴一点、Y轴一点，4代表生成坐标系。
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 控制器只能存储十个工作坐标系，超过十个控制器不予响应，请在标定前查询已有工作坐标系
    RM_APISHARED_EXPORT int Auto_Set_Work_Frame_Cpp(const char *name, byte point_num, bool block);

    ///
    /// \brief Manual_Set_Work_Frame_Cpp 手动设置工作坐标系
    /// \param name 工作坐标系名称，不能超过十个字节。
    /// \param pose 新工作坐标系相对于基坐标系的位姿
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 控制器只能存储十个工作坐标系，超过十个控制器不予响应，请在标定前查询已有工作坐标系
    RM_APISHARED_EXPORT int Manual_Set_Work_Frame_Cpp(const char *name, POSE pose, bool block);

    ///
    /// \brief Change_Work_Frame_Cpp 切换当前工作坐标系
    /// \param name 目标工作坐标系名称
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Change_Work_Frame_Cpp(const char *name, bool block);

    ///
    /// \brief Delete_Work_Frame_Cpp 删除指定工作坐标系
    /// \param name 要删除的工具坐标系名称
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 备注：删除坐标系后，机械臂将切换到机械臂基坐标系
    RM_APISHARED_EXPORT int Delete_Work_Frame_Cpp(const char *name, bool block);

    ////
    /// \brief Get_Current_Work_Frame_Cpp 获取当前工作坐标系
    /// \param frame 返回的坐标系
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Current_Work_Frame_Cpp(FRAME *frame);

    ///
    /// \brief Get_Given_Work_Frame_Cpp 获取指定工作坐标系
    /// \param name 指定的工作坐标系名称
    /// \param pose 获取到的位姿
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Given_Work_Frame_Cpp(const char *name, POSE *pose);

    ///
    /// \brief Get_All_Work_Frame_Cpp 获取所有工作坐标系名称
    /// \param names 返回的工具名称数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_All_Work_Frame_Cpp(FRAME_NAME *names);

    ///
    /// \brief Get_Current_Arm_State_Cpp 获取机械臂当前状态
    /// \param joint 关节1~6角度数组
    /// \param pose 机械臂当前位姿
    /// \param Arm_Err 机械臂运行错误代码
    /// \param Sys_Err 控制器错误代码
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Current_Arm_State_Cpp(float *joint, POSE *pose, uint16_t *Arm_Err, uint16_t *Sys_Err);

    ///
    /// \brief Get_Joint_Temperature_Cpp 获取关节当前温度
    /// \param temperature 关节1~7温度数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Temperature_Cpp(float *temperature);

    ///
    /// \brief Get_Joint_Current_Cpp 获取关节当前电流
    /// \param current 关节1~7电流数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Current_Cpp(float *current);

    ///
    /// \brief Get_Joint_Voltage_Cpp 获取关节当前电压
    /// \param voltage 关节1~7电压数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Voltage_Cpp(float *voltage);

    ///
    /// \brief Get_Joint_Degree_Cpp 获取当前关节角度
    /// \param joint 当前7个关节的角度数组
    /// \return
    ///
    RM_APISHARED_EXPORT int Get_Joint_Degree_Cpp (float *joint);

    ///
    /// \brief Get_Arm_All_State_Cpp 获取机械臂所有状态信息
    /// \param joint_state 存储机械臂信息的结构体
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_All_State_Cpp(JOINT_STATE *joint_state);

    ///
    /// \brief Get_Arm_Plan_Num_Cpp             查询规划计数
    /// \param plan                         查询到的规划计数
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Plan_Num_Cpp(int* plan);

    ///
    /// \brief Set_Arm_Init_Pose_Cpp 设置机械臂的初始位置角度
    /// \param target 机械臂初始位置关节角度数组
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Init_Pose_Cpp(const float *target, bool block);

    ///
    /// \brief Get_Arm_Init_Pose_Cpp 获取机械臂初始位置角度
    /// \param joint 机械臂初始位置关节角度数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Init_Pose_Cpp(float *joint);

    ///
    /// \brief Set_Install_Pose_Cpp     设置安装方式参数
    /// \param x                    旋转角
    /// \param y                    俯仰角
    /// \param block                0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                     0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Install_Pose_Cpp(float x,float y,bool block);

    ///
    /// \brief Movej_Cmd_Cpp 关节空间运动
    /// \param joint 目标关节1~7角度数组
    /// \param v 速度比例1~100，即规划速度和加速度占关节最大线转速和加速度的百分比
    /// \param r 轨迹交融半径，目前默认0。
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待机械臂到达位置或者规划失败
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Movej_Cmd_Cpp(const float *joint, byte v, float r, bool block);

    ///
    /// \brief Movel_Cmd_Cpp 笛卡尔空间直线运动
    /// \param pose 目标位姿,位置单位：米，姿态单位：弧度
    /// \param v 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
    /// \param r 轨迹交融半径，目前默认0。
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待机械臂到达位置或者规划失败
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Movel_Cmd_Cpp(POSE pose, byte v, float r, bool block);

    ///
    /// \brief Movec_Cmd_Cpp 笛卡尔空间 圆弧运动
    /// \param pose_via 中间点位姿，位置单位：米，姿态单位：弧度
    /// \param pose_to 终点位姿
    /// \param v 速度比例1~100，即规划速度和加速度占机械臂末端最大角速度和角加速度的百分比
    /// \param r 轨迹交融半径，目前默认0。
    /// \param loop 规划圈数，目前默认0.
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待机械臂到达位置或者规划失败
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Movec_Cmd_Cpp(POSE pose_via, POSE pose_to, byte v, float r, byte loop, bool block);

    ///
    /// \brief Movej_CANFD_Cpp 角度不经规划，直接通过CANFD透传给机械臂
    /// \param joint 关节1~7目标角度数组
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，由于该模式直接下发给机械臂，不经控制器规划，因此
    /// \return 0-成功，1-失败
    /// 只要控制器运行正常并且目标角度在可达范围内，机械臂立即返回成功指令，此时机械臂可能仍在运行；
    /// 若有错误，立即返回失败指令。
    RM_APISHARED_EXPORT int Movej_CANFD_Cpp(const float *joint, bool block);

    ///
    /// \brief Movep_CANFD_Cpp 位姿不经规划，直接通过CANFD透传给机械臂
    /// \param pose 位姿
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，由于该模式直接下发给机械臂，不经控制器规划，因此
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Movep_CANFD_Cpp(POSE pose, bool block);

    ///
    /// \brief Move_Stop_Cmd_Cpp 突发状况，机械臂已最快速度急停，轨迹不可恢复
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Move_Stop_Cmd_Cpp(bool block);

    ///
    /// \brief Move_Pause_Cmd_Cpp 轨迹暂停，暂停在规划轨迹上，轨迹可恢复
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Move_Pause_Cmd_Cpp(bool block);

    ///
    /// \brief Move_Continue_Cmd_Cpp 轨迹暂停后，继续当前轨迹运动
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Move_Continue_Cmd_Cpp(bool block);

    ///
    /// \brief Clear_Current_Trajectory_Cpp 清除当前轨迹，必须在暂停后使用，否则机械臂会发生意外！！！！
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_Current_Trajectory_Cpp(bool block);

    ///
    /// \brief Clear_All_Trajectory_Cpp 清除所有轨迹，必须在暂停后使用，否则机械臂会发生意外！！！！
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_All_Trajectory_Cpp(bool block);

    ///
    /// \brief Get_Current_Trajectory_Cpp 获取当前正在规划的轨迹信息
    /// \param type 返回的规划类型
    /// \param data 无规划和关节空间规划为当前关节1~7角度数组；笛卡尔空间规划则为当前末端位姿
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Current_Trajectory_Cpp(ARM_CTRL_MODES *type, float *data);

    ///
    /// \brief Add_Waypoint_Cpp 向机械臂添加路径点，机械臂做多可缓存5000个路径点，待收到开始指令后方可运行。
    /// \param joint 目标路径点数组，关节1~6的目标角度，单位：度
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Add_Waypoint_Cpp(const float *joint, bool block);

    ///
    /// \brief Start_Waypoint_Cpp 该函数用于开始运行已经下发的路径点，运行频率按照函数中设置。
    /// \param frq 路径点运行频率，最高不超过200Hz，最低50Hz，否则会发生关节速度不连续的情况
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 注意：开始运行缓存内的轨迹点时，确保当前关节角度与第一个路径点的角度差值不超过3°，否则会报错，该项设置为了保证运行时机械臂出现剧烈抖动。
    RM_APISHARED_EXPORT int Start_Waypoint_Cpp(int frq, bool block);

    ///
    /// \brief Clear_Waypoint_Cpp 清空控制器中所有缓存的路径点
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_Waypoint_Cpp(bool block);

    ///
    /// \brief Stop_Waypoint_Cpp 停止正在运行的轨迹点
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_Waypoint_Cpp(bool block);

    ///
    /// \brief Movej_P_Cmd_Cpp 该函数用于关节空间运动到目标位姿
    /// \param pose: 目标位姿，位置单位：米，姿态单位：弧度。注意：该目标位姿必须是机械臂末端末端法兰中心基于基坐标系的位姿！！
    /// \param v: 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
    /// \param r: 轨迹交融半径，目前默认0。
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待机械臂到达位置或者规划失败
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Movej_P_Cmd_Cpp(POSE pose, byte v, float r, bool block);

    ///
    /// \brief Timer_Cpp        轨迹等待
    /// \param time         等待时间，单位：ms
    /// \param block        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Timer_Cpp(int time,bool block);

    ///
    /// \brief Joint_Teach_Cmd_Cpp 关节示教
    /// \param num 示教关节的序号，1~7
    /// \param direction 示教方向，0-负方向，1-正方向
    /// \param v 速度比例1~100，即规划速度和加速度占关节最大线转速和加速度的百分比
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Joint_Teach_Cmd_Cpp(byte num, byte direction, byte v, bool block);

    ///
    /// \brief Pos_Teach_Cmd_Cpp 当前工作坐标系下，笛卡尔空间位置示教
    /// \param type 示教类型
    /// \param direction 示教方向，0-负方向，1-正方向
    /// \param v 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Pos_Teach_Cmd_Cpp(POS_TEACH_MODES type, byte direction, byte v, bool block);

    ///
    /// \brief Ort_Teach_Cmd_Cpp 当前工作坐标系下，笛卡尔空间位置示教
    /// \param type 示教类型
    /// \param direction 示教方向，0-负方向，1-正方向
    /// \param v 速度比例1~100，即规划速度和加速度占机械臂末端最大角速度和角加速度的百分比
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Ort_Teach_Cmd_Cpp(ORT_TEACH_MODES type, byte direction, byte v, bool block);

    ///
    /// \brief Teach_Stop_Cmd_Cpp 示教停止
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Teach_Stop_Cmd_Cpp(bool block);

    ///
    /// \brief Joint_Step_Cmd_Cpp 关节步进
    /// \param num 关节序号，1~7
    /// \param step 步进的角度，
    /// \param v 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待机械臂返回失败或者到达位置指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Joint_Step_Cmd_Cpp(byte num, float step, byte v, bool block);

    ///
    /// \brief Pos_Step_Cmd_Cpp 当前工作坐标系下，位置示教
    /// \param type 示教类型
    /// \param step 步进的距离，单位m，精确到0.001mm
    /// \param v 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待机械臂返回失败或者到达位置指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Pos_Step_Cmd_Cpp(POS_TEACH_MODES type, float step, byte v, bool block);

    ///
    /// \brief Ort_Step_Cmd_Cpp 当前工作坐标系下，姿态示教
    /// \param type 示教类型
    /// \param step 步进的弧度，单位rad，精确到0.001rad
    /// \param v 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待机械臂返回失败或者到达位置指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Ort_Step_Cmd_Cpp(ORT_TEACH_MODES type, float step, byte v, bool block);

    ///
    /// \brief Get_Controller_State_Cpp 获取控制器状态
    /// \param voltage 返回的电压
    /// \param current 返回的电流
    /// \param temperature 返回的温度
    /// \param sys_err 控制器运行错误代码
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Controller_State_Cpp(float *voltage, float *current, float *temperature, uint16_t *sys_err);

    ///
    /// \brief Set_WiFi_AP_Data_Cpp 控制器WiFi AP模式设置
    /// \param wifi_name 控制器wifi名称
    /// \param password wifi密码
    /// \return 返回值：0-成功，1-失败
    /// 非阻塞模式，下发后，机械臂进入WIFI AP通讯模式
    RM_APISHARED_EXPORT int Set_WiFi_AP_Data_Cpp(const char *wifi_name, const char* password);

    ///
    /// \brief Set_WiFI_STA_Data_Cpp 控制器WiFi STA模式设置
    /// \param router_name 路由器名称
    /// \param password 路由器Wifi密码
    /// \return 0-成功，1-失败
    /// 非阻塞模式：设置成功后，机械臂进入WIFI STA通信模式
    RM_APISHARED_EXPORT int Set_WiFI_STA_Data_Cpp(const char *router_name,const char* password);

    ///
    /// \brief Set_USB_Data_Cpp 控制器UART_USB接口波特率设置
    /// \param baudrate 波特率：9600,38400,115200和460800，若用户设置其他数据，控制器会默认按照460800处理。
    /// \return 0-成功，1-失败
    /// 非阻塞模式，设置成功后机械臂通过UART-USB接口对外通信
    RM_APISHARED_EXPORT int Set_USB_Data_Cpp(int baudrate);

    ///
    /// \brief Set_RS485_Cpp 控制器RS485接口波特率设置
    /// \param baudrate 波特率：9600,38400,115200和460800，若用户设置其他数据，控制器会默认按照460800处理。
    /// \return 0-成功，1-失败
    /// 非阻塞模式，设置成功后机械臂通过UART-USB接口对外通信
    RM_APISHARED_EXPORT int Set_RS485_Cpp(int baudrate);

    ///
    /// \brief Set_Ethernet_Cpp 切换到以太网口工作模式
    /// \return 0-成功，1-失败
    /// 非阻塞模式，设置成功后机械臂通过Ethernet接口对外通信
    RM_APISHARED_EXPORT int Set_Ethernet_Cpp();

    ///
    /// \brief Set_Arm_Power_Cpp 设置机械臂电源
    /// \param cmd true-上电，   false-断电
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Arm_Power_Cpp(bool cmd, bool block);

    ///
    /// \brief Get_Arm_Power_State_Cpp      读取机械臂电源状态
    /// \param power                    获取到的机械臂电源状态
    /// \param block                    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                         0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Power_State_Cpp(int* power);

    ///
    /// \brief Clear_System_Err_Cpp 清除系统错误代码
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_System_Err_Cpp(bool block);


    ///
    /// \brief Get_Arm_Hardware_Version_Cpp     读取硬件版本号
    /// \param version                      获取到的硬件版本号
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Hardware_Version_Cpp(char* version);

    ///
    /// \brief Get_Arm_Software_Version_Cpp     读取软件版本号
    /// \param version                      读取到的用户接口内核版本号
    /// \param ctrl_version                 实时内核版本号
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Arm_Software_Version_Cpp(char* plan_version,char* ctrl_version);

    ///
    /// \brief Get_Log_File_Cpp                 读取SD卡中的Log文件
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Log_File_Cpp(bool block);

    ///
    /// \brief Get_System_Runtime_Cpp           读取控制器的累计运行时间
    /// \param state                        读取结果
    /// \param day                          读取到的时间
    /// \param hour                         读取到的时间
    /// \param min                          读取到的时间
    /// \param sec                          读取到的时间
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_System_Runtime_Cpp(char* state,int* day,int* hour,int* min,int* sec);

    ///
    /// \brief Clear_System_Runtime_Cpp         清零控制器的累计运行时间
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_System_Runtime_Cpp(bool block);

    ///
    /// \brief Get_Joint_Odom_Cpp               读取关节的累计转动角度
    /// \param state                        读取结果
    /// \param odom                         读取到的角度值
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Joint_Odom_Cpp(char* state,float* odom);

    ///
    /// \brief Clear_Joint_Odom_Cpp             清零关节的累计转动角度
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_Joint_Odom_Cpp(bool block);

    ///
    /// \brief Set_High_Speed_Eth_Cpp 设置高速网口
    /// \param num      0-关闭  1-开启
    /// \param block block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    /// 备注：高速网口的IP地址为：192.168.1.18，端口号为：8080，用户不能修改。
    RM_APISHARED_EXPORT int Set_High_Speed_Eth_Cpp(byte num, bool block);

    ///
    /// \brief Set_IO_State_Cpp 设置数字IO状态
    /// \param IO 0-数字IO  1-模拟IO
    /// \param num 通道号，1~4
    /// \param state true-高，   false-低
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_IO_State_Cpp(int IO,byte num, bool state, bool block);

    ///
    /// \brief Get_IO_State_Cpp 获取IO状态
    /// \param IO 0-查询数字IO输出状态 1-查询数字IO输入状态 2-查询模拟IO输出状态 3-查询模拟IO输入状态
    /// \param num
    /// \param state
    /// \return
    ///
    RM_APISHARED_EXPORT int Get_IO_State_Cpp(int IO,byte num, byte *state);

    ///
    /// \brief Get_IO_Input_Cpp 查询所有数字和模拟IO的输入状态
    /// \param DI_state 数字IO输入通道1~3状态数组地址，1-高，0-低
    /// \param AI_voltage 模拟IO输入通道1~4输入电压数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_IO_Input_Cpp(byte *DI_state, float *AI_voltage);

    ///
    /// \brief Get_IO_Output_Cpp 查询所有数字和模拟IO的输出状态
    /// \param DO_state 数字IO输出通道1~4状态数组地址，1-高，0-低
    /// \param AO_voltage 模拟IO输出通道1~4输处电压数组
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_IO_Output_Cpp(byte *DO_state, float *AO_voltage);

    ///
    /// \brief Set_Tool_DO_State_Cpp 设置工具端数字IO输出
    /// \param num 通道号，1~2
    /// \param state true-高，   false-低
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Tool_DO_State_Cpp(byte num, bool state, bool block);

    ///
    /// \brief Set_Tool_IO_Mode_Cpp 设置数字IO模式
    /// \param num 通道号，1~2
    /// \param state 0输入，   1输出
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Tool_IO_Mode_Cpp(byte num, bool state, bool block);

    ///
    /// \brief Get_Tool_IO_State_Cpp 获取数字IO状态
    /// \param IO_Mode 0-输入模式，1-输出模式
    /// \param IO_State 0-低，1-高
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Tool_IO_State_Cpp(float* IO_Mode, float* IO_State);

    ///
    /// \brief Set_Tool_Voltage_Cpp 设置工具端电压输出
    /// \param type 电源输出类型，0-0V，1-5V，2-12V，3-24V
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    int Set_Tool_Voltage_Cpp(byte type, bool block);

    ///
    /// \brief Get_Tool_Voltage_Cpp 查询工具端电压输出
    /// \param voltage 电源输出类型，0-0V，1-5V，2-12V，3-24V
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Tool_Voltage_Cpp(byte *voltage);

    ///
    /// \brief Set_Gripper_Route_Cpp 设置手爪行程
    /// \param min_limit 手爪最小开口，范围 ：0~1000，无单位量纲 无
    /// \param max_limit 手爪最大开口，范围 ：0~1000，无单位量纲 无
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Gripper_Route_Cpp(int min_limit, int max_limit, bool block);

    ///
    /// \brief Set_Gripper_Release_Cpp 手爪松开
    /// \param speed 手爪松开速度 ，范围 1~1000，无单位量纲
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Gripper_Release_Cpp(int speed, bool block);

    ///
    /// \brief Set_Gripper_Pick_Cpp 手爪力控夹取
    /// \param speed 手爪夹取速度 ，范围 1~1000，无单位量纲 无
    /// \param force 力控阈值 ，范围 ：50~1000，无单位量纲 无
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Gripper_Pick_Cpp(int speed, int force, bool block);

    ///
    /// \brief Set_Gripper_Pick_On_Cpp 手爪力控持续夹取
    /// \param ArmSocket socket句柄
    /// \param speed 手爪夹取速度 ，范围 1~1000，无单位量纲 无
    /// \param force 力控阈值 ，范围 ：50~1000，无单位量纲 无
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Gripper_Pick_On_Cpp(int speed, int force, bool block);

    ///
    /// \brief Set_Gripper_Position_Cpp 设置手爪开口度
    /// \param position 手爪开口位置 ，范围 ：1~1000，无单位量纲 无
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Gripper_Position_Cpp(int position, bool block);

    ///
    /// \brief Start_Drag_Teach_Cpp 开始控制机械臂进入拖动示教模式
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Start_Drag_Teach_Cpp(bool block);

    ///
    /// \brief Stop_Drag_Teach_Cpp 控制机械臂退出拖动示教模式
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_Drag_Teach_Cpp(bool block);

    ///
    /// \brief Run_Drag_Trajectory_Cpp 控制机械臂复现拖动示教的轨迹，必须在拖动示教结束后才能使用，
    ///                            同时保证机械臂位于拖动示教的起点位置。
    ///                            若当前位置没有位于轨迹复现起点，请先调用Drag_Trajectory_Origin，否则会返回报错信息。
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Run_Drag_Trajectory_Cpp(bool block);

    ///
    /// \brief Pause_Drag_Trajectory_Cpp 控制机械臂在轨迹复现过程中的暂停
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Pause_Drag_Trajectory_Cpp(bool block);

    ///
    /// \brief Continue_Drag_Trajectory 控制机械臂在轨迹复现过程中暂停之后的继续，
    ///                                 轨迹继续时，必须保证机械臂位于暂停时的位置，
    ///                                 否则会报错，用户只能从开始位置重新复现轨迹。
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Continue_Drag_Trajectory_Cpp(bool block);

    ///
    /// \brief Stop_Drag_Trajectory_Cpp 控制机械臂在轨迹复现过程中的停止
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_Drag_Trajectory_Cpp(bool block);

    ///
    /// \brief Drag_Trajectory_Origin_Cpp 轨迹复现前，必须控制机械臂运动到轨迹起点，
    ///                               如果设置正确，机械臂将以20%的速度运动到轨迹起点
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Drag_Trajectory_Origin_Cpp(bool block);

    ///
    /// \brief Start_Multi_Drag_Teach_Cpp       开始复合模式拖动示教
    /// \param mode                         拖动示教模式 0-电流环模式，1-使用末端六维力，只动位置，2-使用末端六维力 ，只动姿态，3-使用末端六维力，位置和姿态同时动
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Start_Multi_Drag_Teach_Cpp(int mode,bool block);

    ///
    /// \brief Set_Force_Postion_Cpp            力位混合控制
    /// \param mode                         1-基坐标系Z轴力控；2-作用面法线力控；3-作用面径向力控
    /// \param force_level                  力控等级，等级越高，力越小
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Force_Postion_Cpp(int mode,int force_level,bool block);

    ///
    /// \brief Stop_Force_Postion_Cpp           结束力位混合控制
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_Force_Postion_Cpp(bool block);

    ///
    /// \brief Get_Force_Data_Cpp 查询当前六维力传感器得到的力和力矩信息，若要周期获取力数据
    ///                       周期不能小于50ms。
    /// \param Force 返回的力和力矩数组地址，数组6个元素，依次为Fx，Fy，Fz, Mx, My, Mz。
    ///              其中，力的单位为N；力矩单位为Nm。
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Force_Data_Cpp(float *Force);

    ///
    /// \brief Clear_Force_Data_Cpp 将六维力数据清零，即后续获得的所有数据都是基于当前数据的偏移量
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_Force_Data_Cpp(bool block);

    ///
    /// \brief Set_Force_Sensor_Cpp 设置六维力重心参数，六维力重新安装后，必须重新计算六维力所收到的初始力和重心。
    ///                         该指令下发后，机械臂以20%的速度运动到各标定点，该过程不可中断，中断后必须重新标定。
    ///                         重要说明：必须保证在机械臂静止状态下标定。
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Force_Sensor_Cpp(bool block);

    ///
    /// \brief Manual_Set_Force_Cpp 手动设置六维力重心参数，六维力重新安装后，必须重新计算六维力所收到的初始力和重心。
    /// \param type 点位；1~4，调用此函数四次
    /// \param joint 关节角度
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Manual_Set_Force_Cpp(int type,const float* joint);

    ///
    /// \brief Stop_Set_Force_Sensor_Cpp 在标定六维力过程中，如果发生意外，发送该指令，停止机械臂运动，退出标定流程
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_Set_Force_Sensor_Cpp(bool block);

    ///
    /// \brief Set_Hand_Posture_Cpp 设置灵巧手目标手势
    /// \param posture_num 预先保存在灵巧手内的手势序号，范围：1~40
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Hand_Posture_Cpp(int posture_num, bool block);

    ///
    /// \brief Set_Hand_Seq_Cpp 设置灵巧手目标动作序列
    /// \param seq_num 预先保存在灵巧手内的动作序列序号，范围：1~40
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Hand_Seq_Cpp(int seq_num, bool block);

    ///
    /// \brief Set_Hand_Angle_Cpp 设置灵巧手各关节角度
    /// \param angle 手指角度数组，6个元素分别代表6个自由度的角度。范围：0~1000.另外，-1代表该自由度不执行任何操作，保持当前状态
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Hand_Angle_Cpp(const int *angle, bool block);

    ///
    /// \brief Set_Hand_Speed_Cpp 设置灵巧手各关节速度
    /// \param speed 灵巧手各关节速度设置，范围：1~1000
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Hand_Speed_Cpp(int speed, bool block);

    ///
    /// \brief Set_Hand_Force_Cpp 设置灵巧手各关节力阈值
    /// \param force 灵巧手各关节力阈值设置，范围：1~1000，代表各关节的力矩阈值（四指握力0~10N，拇指握力0~15N）。
    /// \param block 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Hand_Force_Cpp(int force, bool block);

    ///
    /// \brief Set_PWM_Cpp 该函数用于设置末端PWM输出
    /// \param Frq: PWM输出频率，范围：100~10000Hz
    /// \param Dpulse: PWM输出占空比，范围：0~100，精度不超过1%
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_PWM_Cpp (int Frq, int Dpulse, bool block);

    ///
    /// \brief Stop_PWM_Cpp 该函数用于停止末端PWM输出
    /// \param block:0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_PWM_Cpp (bool block);

    ///
    /// \brief Get_Fz_Cpp 该函数用于查询末端一维力数据
    /// \param 反馈的一维力数据
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Fz_Cpp (float *Fz);

    ///
    /// \brief Clear_Fz_Cpp 该函数用于清零末端一维力数据
    /// \param 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Clear_Fz_Cpp (bool block);

    ///
    /// \brief 该函数用于自动一维力数据
    /// \param 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Auto_Set_Fz_Cpp(bool block);

    ///
    /// \brief Manual_Set_Fz_Cpp 该函数用于手动设置一维力数据
    /// \param 点位数  0~4
    /// \param 关节角度
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Manual_Set_Fz_Cpp(const float* joint1,const float* joint2);

    ///
    /// \brief Set_Modbus_Mode_Cpp 配置通讯端口 Modbus RTU 模式
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param baudrate: 波特率，支持 9600,115200,460800 三种常见波特率
    /// \param timeout: 超时时间，单位秒。
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Modbus_Mode_Cpp(int port,int baudrate,int timeout,bool block);

    ///
    /// \brief Close_Modbus_Mode_Cpp 关闭通讯端口 Modbus RTU 模式
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Close_Modbus_Mode_Cpp(int port,bool block);

    ///
    /// \brief Get_Read_Coils_Cpp 读线圈
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param address: 线圈起始地址
    /// \param num: 要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
    /// \param device: 外设设备地址
    /// \param coils_data: 返回离散量
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Read_Coils_Cpp(int port,int address,int num,int device,int *coils_data);

    ///
    /// \brief Get_Read_Input_Status_Cpp 读离散量输入
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param address: 线圈起始地址
    /// \param num: 要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
    /// \param device: 外设设备地址
    /// \param coils_data: 返回离散量
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Read_Input_Status_Cpp(int port,int address,int num,int device,int* coils_data);

    ///
    /// \brief Get_Read_Holding_Registers_Cpp 读保持寄存器
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param address: 线圈起始地址
    /// \param device: 外设设备地址
    /// \param coils_data: 返回离散量
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Read_Holding_Registers_Cpp(int port,int address,int device,int* coils_data);

    ///
    /// \brief Get_Read_Input_Registers_Cpp读输入寄存器
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param address: 线圈起始地址
    /// \param device: 外设设备地址
    /// \param coils_data: 返回离散量
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Read_Input_Registers_Cpp(int port,int address,int device,int* coils_data);

    ///
    /// \brief Write_Single_Coil_Cpp写单圈数据
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param address: 线圈起始地址
    /// \param data: 要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
    /// \param device: 外设设备地址
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Write_Single_Coil_Cpp(int port,int address,int data,int device,bool block);

    ///
    /// \brief Write_Coils_Cpp写多圈数据
    /// \param port: 通讯端口，0-控制器RS485端口，1-末端接口板RS485接口
    /// \param address: 线圈起始地址
    /// \param num: 写线圈个数，每次写的数量不超过160个
    /// \param data: 要写入线圈的数据数组，类型：byte。若线圈个数不大于8，则写入的数据为1个字节；否则，则为多个数据的数组
    /// \param device: 外设设备地址
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Write_Coils_Cpp(int port,int address,int num, byte * coils_data,int device,bool block);

    ///
    /// \brief Write_Single_Register_Cpp 写单个寄存器
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param address: 线圈起始地址
    /// \param data: 要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
    /// \param device: 外设设备地址
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Write_Single_Register_Cpp(int port,int address,int data,int device,bool block);

    ///
    /// \brief Write_Registers_Cpp 写多个寄存器
    /// \param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口
    /// \param address: 寄存器起始地址
    /// \param num: 写寄存器个数，寄存器每次写的数量不超过10个
    /// \param data: 要写入寄存器的数据数组，类型：byte
    /// \param device: 外设设备地址
    /// \param block: 0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return 0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Write_Registers_Cpp(int port,int address,int num, byte *single_data, int device, bool block);

    ///
    /// \brief Set_Vehicle_Cpp                  设置移动平台运动
    /// \param speed                        线速度，前进为+，后退为-，精度0.1m/s，最大速度1.5m/s
    /// \param angular                      角速度，左转为+，右转为-，精度0.1rad/s,最大速度2.0rad/s
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Vehicle_Cpp(int speed,int angular,bool block);

    ///
    /// \brief Set_Lift_Cpp                     设置升降平台高度
    /// \param height                       设置高度，单位mm
    /// \param speed                        导轨运行速度百分比，1~100
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Lift_Cpp(int height,int speed,bool block);

    ///
    /// \brief Set_Lift_Speed_Cpp           速度开环控制
    /// \param speed                    speed-速度百分比，-100 ~100
    /// \param block                    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                         0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Lift_Speed_Cpp(int speed,bool block);

    /// \brief Set_Lift_Height_Cpp          位置闭环控制
    /// \param height                   目标高度，单位mm，范围：0~2600
    /// \param speed                    速度百分比，1~100
    /// \param block                    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                         0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Lift_Height_Cpp(int height,int speed,bool block);

    ///
    /// \brief Get_Lift_State_Cpp           获取升降机构状态
    /// \param height                   当前升降机构高度，单位：mm，精度：1mm，范围：0~2300
    /// \param current                  当前升降驱动电流，单位：mA，精度：1mA
    /// \param err_flag                 升降驱动错误代码，错误代码类型参考关节错误代码
    /// \param block                    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                         0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Lift_State_Cpp(int* height,int* current,int* err_flag);

    ///
    /// \brief Set_Car_Move_Cpp             速度开环控制
    /// \param linear                   车体线速度百分比，范围-100~100，>0前进，<0后退
    /// \param angular                  车体旋转速度百分比，范围-100~100，>0逆时针旋转，<0顺时针旋转
    /// \param block                    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                         0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Car_Move_Cpp(int linear,int angular,bool block);

    ///
    /// \brief Set_Car_Station_Cpp          站点导航控制
    /// \param station                  目标站点序号，范围由用户设置
    /// \param speed                    速度百分比，1~100
    /// \param block                    0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                         0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Car_Station_Cpp(int station,int speed,bool block);

    ///
    /// \brief Set_Car_Navigation_Speed_Cpp     速度导航控制
    /// \param speed                        导航速度，-100~100，>0前进，<0后退,0-导航停止
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Car_Navigation_Speed_Cpp(int speed,bool block);

    ///
    /// \brief Set_Car_Navigation_Mode_Cpp      设置导航模式
    /// \param mode                         导航方式，0-磁条导航，1-开环控制模式
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Car_Navigation_Mode_Cpp(int mode,bool block);

    ///
    /// \brief Get_Car_State_Cpp                查询车体状态
    /// \param volume                       反馈车体状态:电池容量，0~100
    /// \param voltage                      反馈车体状态:电池电压，精度0.001V
    /// \param current                      反馈车体状态:电池电压，精度0.001A，>0放电，<0充电
    /// \param car_err                      反馈车体状态:车体运行错误代码，0-正常
    /// \param car_state                    反馈车体状态:车体目前运行状态，0-空闲，1-运动，2-充电
    /// \param station                      反馈车体状态:当前站点序号
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Get_Car_State_Cpp(int* volume,int* voltage,int* current,int* car_err,int* car_state,int* station);

    ///
    /// \brief Set_Car_Charge_Cpp               控制车体自动充电
    /// \param charge_state                 0-解决自动充电模式，1-开始自动充电
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Car_Charge_Cpp(int charge_state,bool block);

    ///
    /// \brief Set_Car_Station_Num_Cpp          设置车体当前站点号
    /// \param charge_state                 站点号
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Set_Car_Station_Num_Cpp(int charge_state,bool block);

    ///
    /// \brief Start_Force_Position_Move_Cpp    开启透传力位混合控制补偿模式
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Start_Force_Position_Move_Cpp(bool block);

    ///
    /// \brief Force_Position_Move_POSE_Cpp          透传力位混合补偿
    /// \param pose                         当前坐标系下目标位姿
    /// \param joint                        目标关节角度
    /// \param sensor                       所使用传感器类型，0-一维力，1-六维力
    /// \param mode                         模式，0-沿基坐标系，1-沿工具端坐标系
    /// \param dir                          力控方向，0~5分别代表X/Y/Z/Rx/Ry/Rz，其中一维力类型时默认方向为Z方向
    /// \param force                        力的大小
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Force_Position_Move_POSE_Cpp(POSE pose,byte sensor,byte mode,int dir,float force,bool block);
    RM_APISHARED_EXPORT int Force_Position_Move_Joint_Cpp(const float *joint,byte sensor,byte mode,int dir,float force,bool block);

    ///
    /// \brief Stop_Force_Postion_Move_Cpp          停止透传力位混合控制补偿模式
    /// \param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
    /// \return                             0-成功，1-失败
    ///
    RM_APISHARED_EXPORT int Stop_Force_Postion_Move_Cpp(bool block);
private:
    SOCKHANDLE m_sockethandle;
};

#endif // RM_API_CPP_H
