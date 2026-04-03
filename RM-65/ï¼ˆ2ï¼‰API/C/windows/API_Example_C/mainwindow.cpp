#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_Start_clicked()
{
    // 连接服务器 返回全局句柄
    m_sockhand = Arm_Socket_Start((char *)"192.168.1.18", 8080, 5000);
    if(m_sockhand <= 0 || m_sockhand > 100000)
    {
        // 连接失败 打印日志
        ui->textEdit->append("socket connect err: " + QString::number(m_sockhand));
        return;
    }
    // 连接成功 打印日志
    ui->textEdit->append("socket connect success: " + QString::number(m_sockhand));
}

void MainWindow::on_pushButton_Test_clicked()
{
    RM_API(65);
    // socket 句柄判断
    if(m_sockhand <= 0)
    {
        // 无效句柄
        ui->textEdit->append("socket is not connect");
        return;
    }

    int nRet = -1;
#if 0
    nRet = Set_Modbus_Mode(m_sockhand, 0, 115200, 300, true);
    if(nRet != 0)
    {
        // 设置失败 打印日志
        ui->textEdit->append("Set_Modbus_Mode err: " + QString::number(nRet));
        return;
    }
    // 设置成功打印日志
    ui->textEdit->append("Set_Modbus_Mode success: " + QString::number(nRet));

    JOINT_STATE joint_states[6];
    memset(joint_states, 0, sizeof(joint_states));
    nRet = Get_Arm_All_State(m_sockhand, joint_states);
    qDebug() << "nRet:" << nRet;



    nRet = Set_IO_State(m_sockhand, 1, 1, true, 0);
    qDebug() << nRet;


    // 设置工具端io输出状态
    nRet = Set_Tool_DO_State(m_sockhand, 1, true, 1);
    qDebug() << nRet;


    // 轨迹复现
    // 进入拖拽模式
    nRet = Start_Drag_Teach(m_sockhand, 1);
    Sleep(3000);

    // 结束拖拽
    nRet = Stop_Drag_Teach(m_sockhand, 1);
    Sleep(3000);

    // 运动到轨迹起点
    Drag_Trajectory_Origin(m_sockhand, 1);
    Sleep(3000);

    // 复现
    Run_Drag_Trajectory(m_sockhand, 1);

    // 力位混合控制
    nRet = Set_Force_Postion(m_sockhand, 1, 1, 1);
    qDebug() << nRet;

    // 设置六维力重心
    nRet = Set_Force_Sensor(m_sockhand, 1);
    qDebug() << nRet;

    // 手动标定六维力数据
    float fJoints[6] = {1,2,3,4,5,6};
    nRet = Manual_Set_Force(m_sockhand, 1, fJoints);
    qDebug() << nRet;

    // 读离散输入量
    int coils_data;
    nRet = Get_Read_Input_Status(m_sockhand, 0, 10, 2, 2, &coils_data);
    qDebug() << nRet;


    nRet = Stop_Force_Postion_Move(m_sockhand, 1);
    qDebug() << nRet;

    #endif
    // 设置安装角度
    byte buf;
    Get_IO_State(m_sockhand, 0, 1, &buf);
}

void MainWindow::on_pushButton_Close_clicked()
{
    // 无需关闭
    if(m_sockhand <= 0)
        return;

    // 关闭连接
    Arm_Socket_Close(m_sockhand);
}
