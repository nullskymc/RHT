import cv2
import numpy as np

def draw_cross(img, point, color, size, thickness=1):
    # 在图像上绘制十字线
    # img: 输入图像
    # point: 十字线的中心点
    # color: 十字线的颜色
    # size: 十字线的大小
    # thickness: 十字线的粗细

    # 绘制横线
    cv2.line(img, (int(point[0] - size / 2), int(point[1])), (int(point[0] + size / 2), int(point[1])), color, thickness)
    # 绘制竖线
    cv2.line(img, (int(point[0]), int(point[1] - size / 2)), (int(point[0]), int(point[1] + size / 2)), color, thickness)

def feature_point(img, img2):
    # 提取图像特征点
    # img: 输入图像
    # img2: 复制的输入图像，用于绘制特征点

    # 将图片转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对灰度图进行高斯模糊处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用Canny算法检测边缘
    edges = cv2.Canny(blurred, 35, 70)
    
    # 复制输入图像
    mm = img2.copy()
    # 将复制的图像设置为全黑色
    mm[:, :] = (0, 0, 0)

    # 查找边缘图像的轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 创建一个全黑的空白图像，用于绘制轮廓
    image_contours = np.zeros(edges.shape, dtype=np.uint8)
    
    # 遍历所有轮廓
    for i in range(len(contours)):
        # 在空白图像上绘制轮廓
        cv2.drawContours(image_contours, contours, i, (255), 1)

    # 遍历所有轮廓
    for contour in contours:
        # 如果轮廓的点数太少或太多，则忽略此轮廓
        if len(contour) <= 100 or len(contour) >= 1000:
            continue
        # 如果轮廓的面积太小或太大，则忽略此轮廓
        if cv2.contourArea(contour) < 10 or cv2.contourArea(contour) > 40000:
            continue
        
        # 如果轮廓的形状不符合要求，则忽略此轮廓
        if not abs((contour[0][0][1] + contour[80][0][1]) / 2 - contour[40][0][1]):
            continue
        
        # 对轮廓进行椭圆拟合
        ellipse = cv2.fitEllipse(contour)
        # 如果椭圆的长短轴比例不符合要求，则忽略此轮廓
        if ellipse[1][0] / ellipse[1][1] < 0.3:
            continue

        # 输出轮廓的面积和点数
        print(f"1：{cv2.contourArea(contour)}")
        print(f"2：{len(contour)}")
        # 在复制的图像上绘制椭圆
        cv2.ellipse(mm, ellipse, (255, 255, 255), -1)

    # 对复制的图像进行高斯模糊处理
    blurred_mm = cv2.GaussianBlur(mm, (5, 5), 0)
    # 使用Canny算法检测边缘
    edges_mm = cv2.Canny(blurred_mm, 50, 150)

    # 查找边缘图像的轮廓
    contours, hierarchy = cv2.findContours(edges_mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 创建一个全黑的空白图像，用于绘制轮廓
    image_contours1 = np.zeros(edges_mm.shape, dtype=np.uint8)

    # 遍历所有轮廓
    for i in range(len(contours)):
        # 在空白图像上绘制轮廓
        cv2.drawContours(image_contours1, contours, i, (255), 1)

    # 遍历所有轮廓
    for contour in contours:
        # 如果轮廓的点数太少或太多，则忽略此轮廓
        if len(contour) <= 100 or len(contour) >= 1000:
            continue
        # 如果轮廓的面积太小或太大，则忽略此轮廓
        if cv2.contourArea(contour) < 3000 or cv2.contourArea(contour) > 20000:
            continue
        
        # 对轮廓进行椭圆拟合
        ellipse = cv2.fitEllipse(contour)
        # 在输入图像上绘制椭圆
        cv2.ellipse(img, ellipse, (255, 0, 0))
        # 输出轮廓的面积和点数
        print(f"mianji: {cv2.contourArea(contour)}")
        print(f"dianshu: {len(contour)}")
        # 计算椭圆的中心点
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        # 在输入图像上绘制十字线
        draw_cross(img, center, (255, 0, 0), 30, 2)
        # 输出中心点的坐标
        print(f"{center[0]} {center[1]}")

def main():
    # 主函数
    # 读入图像，提取特征点，显示图像

    # 图像文件名
    imagename = "./testdata/0859.png"
    
    # 从文件中读入图像
    img = cv2.imread(imagename)
    # 复制输入图像
    img2 = cv2.imread(imagename)
    
    # 如果读入图像失败
    if img is None:
        print(f"Can not load image {imagename}")
        return
    
    # 提取图像特征点
    feature_point(img, img2)
    
    # 显示图像
    cv2.imshow("image", img)
    # 等待按键，按键盘任意键就返回
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 如果是直接运行的话，调用主函数
    main()