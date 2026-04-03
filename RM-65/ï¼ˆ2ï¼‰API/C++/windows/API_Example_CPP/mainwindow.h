#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "rm_api_cpp.h"
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
    void on_pushButton_Start_clicked();

    void on_pushButton_Test_clicked();

    void on_pushButton_Close_clicked();

private:
    Ui::MainWindow *ui;

    RM_API_CPP * m_pApi = nullptr;
};

#endif // MAINWINDOW_H
