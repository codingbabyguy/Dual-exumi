#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "rm_api.h"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    // 连接 socket
    void on_pushButton_Start_clicked();

    // 接口测试
    void on_pushButton_Test_clicked();

    // 关闭 socket
    void on_pushButton_Close_clicked();


private:
    Ui::MainWindow *ui;

    // 手动维护句柄
    SOCKHANDLE m_sockhand = -1;
};

#endif // MAINWINDOW_H
