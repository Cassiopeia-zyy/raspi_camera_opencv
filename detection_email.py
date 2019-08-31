import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr


count = 1
timeF = 30
camera = cv2.VideoCapture(0)   # 参数0表示第一个摄像头
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def send_email():
    sender = '965655980@qq.com'  # 邮件发送者的邮箱地址
    receivers = '2482038370@qq.com'  # 邮件接收者的邮箱地址
    subject = 'camera from home'     # 主题内容
    msg = 'find stranger in your home and the pictures have been saved!'   # 邮件内容

    message = MIMEText(msg, 'plain', 'utf-8')   # 三个参数：1.邮件正文 2.文本格式 3.编码格式:utf-8 设置编码
    message['Subject'] = Header(subject, 'utf-8')  # 邮件标题
    message['From'] = formataddr(["Camera", sender])  # 发送者
    message['To'] = formataddr(["client", receivers])  # 接收者

    smtp_server = smtplib.SMTP('smtp.qq.com', port=25)  # 邮件服务器
    smtp_server.login(user=sender, password='vwgmvegwjglabbcc')  # password是授权码
    smtp_server.sendmail(sender, receivers, message.as_string())   # 发送邮件


while True:
    ret, frame = camera.read()
    flat = 0
    fgmask = bs.apply(frame)  # 背景分割器，该函数计算了前景掩码
    # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值，在阈值化图像中，将非纯白色（244~255）的所有像素都设为0，而不是255
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    # 下面就跟基本运动检测中方法相同，识别目标，检测轮廓，在原始帧上绘制检测结果
    dilated = cv2.dilate(th, es, iterations=2)  # 形态学膨胀
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) > 1600:
            flat = 1
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # cv2.imshow('mog', fgmask)
    # cv2.imshow('thresh', th)
    cv2.imshow('detection', frame)
    key = cv2.waitKey(1) & 0xFF
    # 按'ESC'健退出循环
    if key == 27:
        break

    if flat == 1:   # 设置一个标签，当有运动的时候为1
        if (count % timeF == 0):
            path = 'E:/opencv+py/photo_save\shot%d.jpg' % (count)
            cv2.imwrite(path, frame)
            if count == 150:
                send_email()
        count += 1

        continue


# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
