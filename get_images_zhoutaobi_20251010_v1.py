import time
import os
import datetime
import cv2
import keyboard  # 用于监听键盘输入（需额外安装）
from ur5_robot import UR_Robot
from real.realsenseD415 import Camera


def main():
    # 初始化机械臂对象
    robot = UR_Robot(
        robot_ip="192.168.1.35",
        gripper_port=False,
        is_use_robot=True,
        is_use_camera=True
    )

    # 初始化相机
    try:
        camera = Camera(width=640, height=480, fps=30)
        print("相机初始化成功")
    except Exception as e:
        print(f"相机初始化失败: {e}")
        return

    try:
        # 创建带时间戳的文件夹（存储黑白图片和姿态数据）
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"images_{current_time}"
        os.makedirs(folder_name, exist_ok=True)
        print(f"黑白照片和姿态数据将保存到: {folder_name}")

        # 提示操作指令
        print("\n=== 操作说明 ===")
        print("1. 实时显示框将展示相机黑白画面")
        print("2. 按下键盘 'y' 键：拍摄1张照片并记录机械臂姿态")
        print("3. 按下键盘 'q' 键：退出程序")
        print("===============\n")

        photo_count = 0  # 照片计数器
        print("实时显示已启动，等待键盘操作...")

        # 循环实时显示相机画面，监听键盘输入
        while True:
            # 1. 实时获取相机画面并转为黑白（用于显示和后续保存）
            color_image, depth_image = camera.get_data()
            blackwhite_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 2. 在实时画面上添加操作提示文字（便于查看指令）
            # 文字1：操作提示（顶部）
            cv2.putText(
                blackwhite_image,
                "Press 'y' to take photo | Press 'q' to quit",
                (10, 30),  # 文字位置（x,y）
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # 字体大小
                (255, 255, 255),  # 文字颜色（白，因黑白图单通道）
                2  # 文字粗细
            )
            # 文字2：已拍摄照片数量（底部）
            cv2.putText(
                blackwhite_image,
                f"Photos Taken: {photo_count}",
                (10, 460),  # 底部位置（避免遮挡画面）
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # 3. 弹出实时显示框（窗口名：RealSense Live View）
            cv2.imshow("RealSense Live View", blackwhite_image)

            # 4. 监听键盘输入（关键逻辑）
            # 按下 'y' 键：拍照+记录姿态
            if keyboard.is_pressed('y'):
                photo_count += 1
                # 保存黑白照片
                bw_photo_filename = f"{folder_name}/bw_photo_{photo_count:04d}.jpg"
                cv2.imwrite(bw_photo_filename, blackwhite_image)
                # 保存深度图像（可选）
                depth_filename = f"{folder_name}/depth_{photo_count:04d}.png"
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(depth_filename, depth_colormap)
                # 获取并记录机械臂姿态
                tcp_pose = robot.get_actual_tcp_pose()
                joint_positions = robot.get_actual_joint_position()
                pose_file = f"{folder_name}/pose.txt"
                with open(pose_file, "a") as f:
                    f.write(f"{tcp_pose}\n")
                # 控制台提示拍照成功
                print(f"\n[拍照成功] 第{photo_count}张 | 照片：{bw_photo_filename} | 姿态已记录")
                # 短暂延时（避免长按y键连续拍照，可根据需求调整）
                time.sleep(0.3)

            # 按下 'q' 键：退出程序（需先按q，再按回车确认）
            elif keyboard.is_pressed('q'):
                print("\n检测到 'q' 键，准备退出程序...")
                break

            # 5. 窗口刷新延时（确保显示流畅，避免CPU占用过高）
            cv2.waitKey(1)

    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        # 资源清理（关键：关闭窗口和连接）
        cv2.destroyAllWindows()  # 关闭所有OpenCV显示窗口
        robot.rtde_c.stopScript()  # 关闭机械臂连接
        print("\n实时显示窗口已关闭 | 机械臂连接已关闭 | 程序退出完成")


if __name__ == "__main__":
    # 先检查keyboard库是否安装，未安装则提示
    try:
        import keyboard
    except ImportError:
        print("缺少 'keyboard' 库，需先安装：pip install keyboard")
        exit()
    # 启动主程序
    main()