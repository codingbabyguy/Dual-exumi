# This Python file uses the following encoding: utf-8

# if __name__ == "__main__":
    #     pass
import ctypes
import os
import time

if __name__ == "__main__":
    CUR_PATH=os.path.dirname(__file__)
    dllPath=os.path.join(CUR_PATH,"RM_API_C.dll")
    pDll=ctypes.cdll.LoadLibrary(dllPath)

#   设置机械臂关节版本
    pDll.RM_API(6)

#   连接机械臂
    nSocket = pDll.Arm_Socket_Start("192.168.1.17",8080,200)
    print (nSocket)
    
#   查询机械臂连接状态
    nRet = pDll.Arm_Sockrt_State(nSocket);
    print (nRet)
    
#   设置机械臂末端参数为初始值
    nRet = pDll.Set_Arm_Tip_Init(nSocket, 1);
    print (nRet)
    
#   设置机械臂动力学碰撞检测等级
    nRet = pDll.Set_Collision_Stage(nSocket, 1, 1);
    print (nRet)
    
    
    i = 1
    while i < 5:
        time.sleep(1)
        
#   关闭连接
    pDll.Arm_Socket_Close(nSocket);
