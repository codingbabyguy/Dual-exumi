#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    m_pApi = new RM_API_CPP(6);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_Start_clicked()
{
    int nRet = m_pApi->Arm_Socket_Start_Cpp((char*)"192.168.1.200", 8080, 200);
    if(nRet != 0)
    {
        ui->textEdit->append("socket connect err: " + QString::number(nRet));
    }
    ui->textEdit->append("socket connect success: " + QString::number(nRet));
}

void MainWindow::on_pushButton_Test_clicked()
{
    int nRet = m_pApi->Set_Modbus_Mode_Cpp(0, 115200, 300, true);
    if(nRet != 0)
    {
        // 设置失败 打印日志
        ui->textEdit->append("Set_Modbus_Mode err: " + QString::number(nRet));
        return;
    }
    // 设置成功打印日志
    ui->textEdit->append("Set_Modbus_Mode success: " + QString::number(nRet));
}

void MainWindow::on_pushButton_Close_clicked()
{
    // 关闭连接
    m_pApi->Arm_Socket_Close_Cpp();
}
