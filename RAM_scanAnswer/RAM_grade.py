# -*- codeing = utf-8 -*-
# @TIME:2018/5/3 10:27
# @Auther:RAM
# @File:RAM_grade.py
# @Software:PyCharm


# 思路是嘛呢
# 1.预处理图片
# 2.提取最外层轮廓
# 3.提取每个选框
# 4.判断每道题哪个填最满
# 5.判分


import cv2
import numpy as np
from imutils.perspective import four_point_transform  # 用来透视变换

# -------------------------------------------------------------------------
# 0:A,1:B,2:C,3:D
import random  # 懒得设置答案，随机生成
ques_nu = [i for i in range(1, 41)]
ques_answer = [random.randint(0, 3) for i in range(40)]  # 注意randint是闭区间
ANSWER_KEY = dict(zip(ques_nu, ques_answer))
print("答案"+'\n')
print(ANSWER_KEY)

# --------------------------------------------------------------------------
# 为了方便调试，先定义几个好用的函数
def cv_show(name, img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def find_maxcnts(cnts):
	if len(cnts) > 0:
		# 将轮廓按大小降序排序
		cntsall = sorted(cnts, key=cv2.contourArea, reverse=True)  # 以轮廓面积来排序

		# 对排序后的轮廓循环处理
		for c in cntsall:
			# 获取近似的轮廓
			peri = cv2.arcLength(c, True)  # 计算轮廓周长
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 以0.02的公差重新生成近似的轮廓，去毛刺

			# 如果我们的近似轮廓有四个顶点，那么就认为找到了答题卡
			if len(approx) == 4:  # 或者if cv2.isContourConvex(approx):
				docCnt = approx
				horn = cv2.convexHull(approx)  # 这既是四个顶角
				break
	return docCnt


def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	bounding_Box = [cv2.boundingRect(c) for c in cnts]
	(cnts, bounding_Box) = zip(*sorted(zip(cnts, bounding_Box),
									   key=lambda b: b[1][i], reverse=reverse))
	return cnts, bounding_Box


def sift_rect(cnts):
	ques_cnts = []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		if (ar > 1.0 and ar < 2.0) and (w > 30 and w < 42):
			ques_cnts.append(c)
	return ques_cnts


def drawcnts(img, cnts, drawname):
	drawimg = img.copy()
	drawcn = cv2.drawContours(drawimg, cnts, -1, (0, 0, 255), 2)
	cv_show(drawname, drawcn)
# --------------------------------------------------------------------------第一步
# 加载图片，将它转换为灰阶，轻度模糊，然后边缘检测提取二值图。

image = cv2.imread('./origin.png')  # 以数组形式读取存在ndarray中，而PIl读取为图片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)
cv_show('Edged', edged)
prep = np.hstack((gray, blurred, edged))  # 这里要预览的话需要resize函数来调整大小,resize也能用来识别
# cv_show('preprocess', prep)

# -------------------------------------------------------------------------第二步
#

# 从边缘图中寻找轮廓，然后初始化答题卡对应的轮廓
cntsall = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

drawcnts(image.copy(), cntsall, "Allcnts")

# 确保至少有一个轮廓被找到
docCnt = find_maxcnts(cntsall)

# 对原始图像和灰度图都进行四点透视变换
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
# 答题卡可能有内外两层，需要裁剪
## 这里怎么操作能拿到内轮廓呢，迭代没用，遍历找最大也不是最大。。。
(h, w) = warped.shape
warped1 = warped[10:h - 10, 10:w - 10]
paper1 = paper[10:h - 10, 10:w - 10]
cv_show('paper', paper1)
print("定位出答题卡区域")

# ------------------------------------------------------------------------------第三步
# 首先只保留方框，然后按行分类，再按每四个为一组分类


# 首先提取轮廓并筛选，只选出方框图

thresh_paper = cv2.threshold(warped1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv_show('thresh_paper', thresh_paper)
cntsall_rect = cv2.findContours(thresh_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
drawcnts(paper1, cntsall_rect, "All_rect")

ques_cnts = sift_rect(cntsall_rect)
drawcnts(paper1, ques_cnts, 'Question_rect')  # 调试保证留下160个框

# 按行分类，再按每四个一组分类

ques_cnts = sort_contours(ques_cnts,method='top-to-bottom')[0]
correct = 0
ans_paper = paper1.copy()
for (row_num, rect_row) in enumerate(np.arange(0, len(ques_cnts) - 4, 12)):  # 最后一行下个版本再修复，就是一个if筛出来就好
	rect_ans = sort_contours(ques_cnts[rect_row:rect_row+12])[0]
	for (i, rect_num) in enumerate(np.arange(0, 12, 4)):
		rect_ans_per = rect_ans[rect_num:rect_num + 4]
		bubbled = None
		# 判断每道题结果
		for (j, c) in enumerate(rect_ans_per):
			# 使用mask来判断结果
			mask = np.zeros(thresh_paper.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)  # -1表示填充
			# cv_show('mask', mask)

			# 通过计算非零点数量来算是否选择这个答案
			mask = cv2.bitwise_and(thresh_paper, thresh_paper, mask=mask)
			total = cv2.countNonZero(mask)

			# 通过阈值判断
			if bubbled is None or total > bubbled[0]:
				bubbled = (total, j)

		# 对比正确答案
		color = (0, 0, 255)
		ques_num = row_num * 3 + i + 1
		k = ANSWER_KEY[ques_num]

		# 判断正确
		if k == bubbled[1]:
			color = (0, 255, 0)
			correct += 1

		# 绘图
		cv2.drawContours(ans_paper, rect_ans_per, k, color, 3)

# --------------------------------------------------------------第四步:判分
score = (correct / 40.0) * 100
print("最终得分：%.2f分" % score)
cv_show('result', ans_paper)
