import cv2
import numpy as np
import os
import math
from pylsd.lsd import lsd
from scipy.spatial import distance

from modules import *
from cut_page import cut_page

def frame_detect(image):
    """
    画像からコマをルールベースで抽出する関数。
    
    """
    if image is None:
        print("Not open:", image)
        return
    # srcがカラーの場合グレースケールに変換
    if len(image.shape) == 3:
        color_img = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    gaus = cv2.GaussianBlur(image, (3, 3), 0)
    
    lines = lsd(gaus)
    lines_img = np.zeros(image.shape, dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[:4])
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 1000:
            lines_img = cv2.line(lines_img, (x1, y1), (x2, y2), (255), 1, cv2.LINE_AA)
    cv2.imshow('lines_img', lines_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    hough_lines = cv2.HoughLines(lines_img, 1, np.pi / 180.0, 250)
    hough_lines2 = cv2.HoughLines(lines_img, 1, np.pi / 360.0, 250)
    lines_img = np.zeros(image.shape, dtype=np.uint8)
    for rho, theta in hough_lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        lines_img = cv2.line(lines_img, pt1, pt2, (255), 3, cv2.LINE_AA)
    for rho, theta in hough_lines2[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        lines_img = cv2.line(lines_img, pt1, pt2, (255), 3, cv2.LINE_AA)
    cv2.imshow('lines_img', lines_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gaussian_img = cv2.GaussianBlur(image, (3, 3), 0)
    binForSpeechBalloon_img = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)[1]
    binForSpeechBalloon_img = cv2.erode(binForSpeechBalloon_img, np.ones((3, 3), np.uint8), iterations=1)
    binForSpeechBalloon_img = cv2.dilate(binForSpeechBalloon_img, np.ones((3, 3), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    hukidashied_img = extractSpeechBalloon(contours, hierarchy, gaussian_img)
    cv2.imshow('hukidashied_img', hukidashied_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hukidashied_img = cv2.GaussianBlur(hukidashied_img, (3, 3), 0)
    hukidashied_img = cv2.threshold(hukidashied_img, 210, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('hukidashied_img', hukidashied_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    and_img = cv2.bitwise_and(hukidashied_img, lines_img)
    cv2.imshow('and_img', and_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    contours, _ = cv2.findContours(and_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 1. 完全に内包された輪郭を除去する関数
    def remove_inner_contours(contours):
        remaining_contours = []
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        
        for i, box1 in enumerate(bounding_boxes):
            x1, y1, w1, h1 = box1
            is_inner = False
            for j, box2 in enumerate(bounding_boxes):
                if i != j:  # 比較対象が異なる輪郭の場合
                    x2, y2, w2, h2 = box2
                    # box1がbox2に完全に内包されているかを判定
                    if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                        is_inner = True
                        break
            if not is_inner:
                remaining_contours.append(contours[i])
        
        return remaining_contours
    remaining_contours = remove_inner_contours(contours)
    
    # 2. バウンディングボックスの頂点に最も近い輪郭点を求める関数
    def find_closest_points(contour, bounding_box):
        x, y, w, h = bounding_box
        bbox_points = np.array([
            [x, y],         # 左上
            [x + w, y],     # 右上
            [x, y + h],     # 左下
            [x + w, y + h]  # 右下
        ])
        
        contour_points = contour.reshape(-1, 2)
        closest_points = []
        
        for bbox_point in bbox_points:
            distances = distance.cdist([bbox_point], contour_points, metric='euclidean')
            closest_point_idx = np.argmin(distances)
            closest_points.append(tuple(contour_points[closest_point_idx]))
        
        return closest_points
    results = []
    for contour in remaining_contours:
        bounding_box = cv2.boundingRect(contour)
        closest_points = find_closest_points(contour, bounding_box)
        #xが最大かつyが最小は右上，xが最小かつyが最小は左上，xが最大かつyが最大は右下，xが最小かつyが最大は左下
        # top_right,top_left,bottom_right,bottom_leftにそれぞれ格納
        top_right, top_left, bottom_right, bottom_left = None, None, None, None
        for x, y in closest_points:
            if top_right is None or (x >= top_right[0] and y <= top_right[1]):
                top_right = (x, y)
            if top_left is None or (x <= top_left[0] and y <= top_left[1]):
                top_left = (x, y)
            if bottom_right is None or (x >= bottom_right[0] and y >= bottom_right[1]):
                bottom_right = (x, y)
            if bottom_left is None or (x <= bottom_left[0] and y >= bottom_left[1]):
                bottom_left = (x, y)
        results.append([top_right, top_left, bottom_left, bottom_right])
        
    result_img = np.zeros(image.shape, dtype=np.uint8)
    for result in results:
        #resultの4点を結ぶ線を描画
        cv2.polylines(result_img, [np.array(result)], isClosed=True, color=(255), thickness=3)
    cv2.imshow('result_img', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # gaus = cv2.GaussianBlur(hukidashied_img, (3, 3), 0)
    # canny_img = cv2.Canny(gaus, 50, 110, apertureSize=3)
    # cv2.imshow('canny_img', canny_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # hough_lines = cv2.HoughLines(canny_img, 1, np.pi / 180.0, 200)
    # hough_lines2 = cv2.HoughLines(canny_img, 1, np.pi / 360.0, 200)
    # lines_img = np.zeros(image.shape, dtype=np.uint8)
    # for rho, theta in hough_lines[:, 0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    #     pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    #     lines_img = cv2.line(lines_img, pt1, pt2, (255), 3, cv2.LINE_AA)
    # for rho, theta in hough_lines2[:, 0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    #     pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    #     lines_img = cv2.line(lines_img, pt1, pt2, (255), 3, cv2.LINE_AA)
    # cv2.imshow('lines_img', lines_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # and_img = np.zeros(image.shape, dtype=np.uint8)
    # and_img = cv2.bitwise_and(hukidashied_img, lines_img)
    # cv2.imshow('and_img', and_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # #輪郭抽出・ラベリング
    # contours, _ = cv2.findContours(and_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # #輪郭描画
    # result_img = np.zeros(image.shape, dtype=np.uint8)
    # for i in range(len(contours)):
    #     print('contours', i)
    #     cv2.drawContours(result_img, contours, i, (255), 3)
    # cv2.imshow('result_img', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # boundingbox_from_and_img = and_img.copy()

    # complement_and_img = createAndImgWithBoundingBox(and_img, contours, hukidashied_img)
    
    # contours2, _ = cv2.findContours(complement_and_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # binForSpeechBalloon_img = cv2.threshold(src_img, 230, 255, cv2.THRESH_BINARY)[1]
    # # cv2.imshow('binForSpeechBalloon_img', binForSpeechBalloon_img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # # 膨張収縮
    # kernel = np.ones((3, 3), np.uint8)
    # binForSpeechBalloon_img = cv2.erode(binForSpeechBalloon_img, kernel,(-1,-1), iterations = 1)
    # binForSpeechBalloon_img = cv2.dilate(binForSpeechBalloon_img, kernel,(-1,-1), iterations = 1)
    # # cv2.imshow('binForSpeechBalloon_img', binForSpeechBalloon_img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # hierarchy2 = []  # cv::Vec4i のリスト
    # hukidashi_contours = []  # cv::Point のリストのリスト（輪郭情報）

    # # 輪郭抽出
    # hukidashi_contours, hierarchy2 = cv2.findContours(binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0)

    # # 吹き出し検出　塗りつぶし
    # gaussian_img = extractSpeechBalloon(hukidashi_contours, hierarchy2, gaussian_img)
    # cv2.imshow('gaussian_img', gaussian_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # cv2.imwrite('./output/0604/004_gaussian_img.jpg', gaussian_img)


    # # inverse_bin_img = cv2.bitwise_not(binForSpeechBalloon_img)
    # inverse_bin_img = cv2.threshold(gaussian_img,210,255,cv2.THRESH_BINARY_INV)[1]
    # cv2.imwrite('./output/0604/004_inverse_bin_img.jpg', inverse_bin_img)

    # ####################ここまでOK####################


    # pageCorners,_, = findFrameExistenceArea(inverse_bin_img)
    # print('pageCorners',pageCorners) # ここ怪しい

    # canny_img = cv2.Canny(inverse_bin_img, 50, 110, apertureSize=3) # 元の引数は120,130
    # # ここ元は，src_imgに対してCannyをかけていたが，gaussian_imgに対してかけるように変更

    # lines = []
    # lines2 = []

    # lines = cv2.HoughLines(canny_img, 1, np.pi / 180.0, 200)
    # # lines2 = cv2.HoughLines(canny_img, 1, np.pi / 360.0, 250)

    # lines_img = np.zeros(src_img.shape, dtype=np.uint8)

    # lines_img = drawLines(lines, lines_img)
    # lines_img = drawLines(lines2, lines_img)
    # cv2.imwrite('./output/0604/004_lines_img.jpg', lines_img)

    # ####################ここまでOK????####################

    # and_img = np.zeros(src_img.shape, dtype=np.uint8)
    # and_img = cv2.bitwise_and(inverse_bin_img, lines_img)
    # cv2.imwrite('./output/0604/004_and_img.jpg', and_img)

    # ####################ここまでOK####################

    # contours = []
    # tmp_img = and_img.copy()
    # contours, _ = cv2.findContours(tmp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # boundingbox_from_and_img = and_img.copy()

    # complement_and_img = createAndImgWithBoundingBox(boundingbox_from_and_img, contours,inverse_bin_img)
    # cv2.imwrite('./output/0604/004_complement_and_img.jpg', complement_and_img)

    # ####################ここまでOK####################

    # contours3 = []
    # bounding_boxes = []


    # contours3, _ = cv2.findContours(complement_and_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # for i, contour in enumerate(contours3):
    # #     print(f"Contour {i+1}:")
    # #     for j, point in enumerate(contour):
    # #         print(f"Point {j+1}: x = {point[0][0]}, y = {point[0][1]}")
    # # cv2.imshow('contours3',complement_and_img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # for i in range(len(contours3)):
    #     tmp_bounding_box = cv2.boundingRect(contours3[i])

    #     if judgeAreaOfBoundingBox(tmp_bounding_box, complement_and_img.shape[0]*complement_and_img.shape[1]):
    #         bounding_boxes.append(tmp_bounding_box)

    # # print('bounding_boxes',len(bounding_boxes))
    # # cv2.imshow('complement_and_img',complement_and_img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # ####################ここまでOK####################

    # for i in range(len(contours3)):
    #     # print('contours3',contours3[i])
    #     approx = cv2.approxPolyDP(contours3[i], 6, True)
    #     # print('approx_size',len(approx))
    #     # cv2.imshow("approx", complement_and_img)
    #     # cv2.waitKey(0)

    #     # Create a bounding rectangle
    #     brect = cv2.boundingRect(contours3[i])
    #     print(brect)

    #     # Coordinates of the top left and bottom right
    #     xmin = brect[0]
    #     ymin = brect[1]
    #     xmax = brect[0] + brect[2]
    #     ymax = brect[1] + brect[3]

    #     if xmin<6:xmin = 0
    #     if xmax>inverse_bin_img.shape[1]-6:xmax = inverse_bin_img.shape[1]
    #     if ymin<6:ymin = 0
    #     if ymax>inverse_bin_img.shape[0]-6:ymax = inverse_bin_img.shape[0]

    #     bbPoints = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]], dtype=np.int32)
    #     top_line, bottom_line, left_line, right_line = renew_line(bbPoints)

    #     # 大きさ４で初期化
    #     definitePanelPoint = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)

    #     flag_LT = False
    #     flag_LB = False
    #     flag_RT = False
    #     flag_RB = False

    #     bb_min_LT = src_img.shape[0]
    #     bb_min_RT = src_img.shape[0]
    #     bb_min_LB = src_img.shape[0]
    #     bb_min_RB = src_img.shape[0]

    #     isOverlap = True

    #     if judgeAreaOfBoundingBox(brect, src_img.shape[1] * src_img.shape[0]):
    #         # Check if bounding boxes overlap
    #         isOverlap = judgeBoundingBoxOverlap(isOverlap, bounding_boxes, brect)
    #     else:
    #         isOverlap = False

    #     # if not isOverlap:
    #     #     continue
    #     for i in range(len(approx)):
    #         p = approx[i][0]

    #         # print('deffinitePanel',definitePanelPoint)
    #         # print('pagecorners',pageCorners)
    #         # print('bbpoints',bbPoints)

    #         flag_LT, bb_min_LT, definitePanelPoint[0] = definePanelCorners(flag_LT, p, bb_min_LT,  pageCorners[0],  definitePanelPoint[0], bbPoints[0])
    #         flag_LB, bb_min_LB, definitePanelPoint[1] = definePanelCorners(flag_LB, p, bb_min_LB,  pageCorners[1],  definitePanelPoint[1], bbPoints[1])
    #         flag_RT, bb_min_RT, definitePanelPoint[2] = definePanelCorners(flag_RT, p, bb_min_RT,  pageCorners[2],  definitePanelPoint[2], bbPoints[2])
    #         flag_RB, bb_min_RB, definitePanelPoint[3] = definePanelCorners(flag_RB, p, bb_min_RB,  pageCorners[3],  definitePanelPoint[3], bbPoints[3])
    #         definitePanelPoint = align2edge(definitePanelPoint, inverse_bin_img)

    #     top_line, bottom_line, left_line, right_line = renew_line(definitePanelPoint)

    #     alphaImage = cv2.cvtColor(color_img, cv2.COLOR_BGR2BGRA)

    #     # createAlphaImage(alphaImage, definitePanelPoint)

    #     cut_img = alphaImage[brect[1]:brect[1]+brect[3], brect[0]:brect[0]+brect[2]]
    #     # cv2.imshow("cut_img",cut_img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     panel_imgs = []
    #     panel_imgs.append(cut_img)
    #     # print(panel_imgs[0].shape)

    #     # print('definitePanelPoint[0]',definitePanelPoint[0])
    #     # print('definitePanelPoint[1]',definitePanelPoint[1])
    #     # print('definitePanelPoint[2]',definitePanelPoint[2])
    #     # print('definitePanelPoint[3]',definitePanelPoint[3])
    #     cv2.line(color_img,definitePanelPoint[0],definitePanelPoint[2],(255,0,0),thickness=2,lineType=8)
    #     cv2.line(color_img,definitePanelPoint[2],definitePanelPoint[3],(255,0,0),thickness=2,lineType=8)
    #     cv2.line(color_img,definitePanelPoint[3],definitePanelPoint[1],(255,0,0),thickness=2,lineType=8)
    #     cv2.line(color_img,definitePanelPoint[1],definitePanelPoint[0],(255,0,0),thickness=2,lineType=8)

    # print(len(panel_imgs))
    # for i in range (len(panel_imgs)):
    #     cv2.imshow("panel"+str(i),panel_imgs[i])
    #     cv2.imwrite("./output/panel/panel"+str(i)+".png",panel_imgs[i])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()







def extractSpeechBalloon(contours,hierarchy,img):
    # for contour in contours:
    #     if len(contour) == 0:
    #         continue
    #     mask = np.zeros_like(img)
    #     cv2.drawContours(mask, [contour], -1, 255, -1)
    #     roi = cv2.bitwise_and(img, mask)
    #     cv2.imshow('roi', roi)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        
    #     white_pixels = np.sum(roi == 255)
    #     black_pixels = np.sum(roi == 0)
        
    #     # 白黒比を計算
    #     if white_pixels + black_pixels > 0:
    #         ratio = white_pixels / (white_pixels + black_pixels)
    #     else:
    #         ratio = 0  # 万が一、領域内にピクセルがなければ比率を0とする
    #     if ratio > 0.4:
    #         cv2.drawContours(img, [contour], -1, 0, -1, cv2.LINE_AA, hierarchy, 1)
        
    if len(contours) == 0:
        print("Speech balloon not found.")
        raise ValueError
    for i in range(len(contours)):
        # if contours[contours].dtype != np.float32 and contours[i].dtype != np.int32:
        #     contours[i] = contours[i].astype(np.float32)  # 必要に応じて変換
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        en = 0.0
        if img.shape[0] * img.shape[1] * 0.001 <= area and area < img.shape[0] * img.shape[1] * 0.03:
            en = 4.0 * np.pi * area / (length * length)
            if en > 0.3:
                cv2.drawContours(img, contours, i, 0, -1, cv2.LINE_AA, hierarchy, 1)
    return img

#コマが存在しない余白の削除
def remove_margins(binary_image):
    vertical_projection = np.sum(binary_image == 255, axis=1)  # 縦方向の白画素数
    horizontal_projection = np.sum(binary_image == 255, axis=0)  # 横方向の白画素数
    
    # 上下の余白
    top = np.argmax(vertical_projection > 0)
    bottom = len(vertical_projection) - np.argmax(vertical_projection[::-1] > 0)
    
    # 左右の余白
    left = np.argmax(horizontal_projection > 0)
    right = len(horizontal_projection) - np.argmax(horizontal_projection[::-1] > 0)
    
    cropped_image = binary_image[top:bottom, left:right]
    offsets = {"top": top, "left": left}
    return cropped_image, offsets

# 3. 座標補正
def correct_coordinates(coords, offsets):
    """
    coords: [(x1, y1, x2, y2), ...] 形式の座標リスト
    offsets: {"top": int, "left": int} 形式のオフセット
    """
    corrected_coords = []
    for x1, y1, x2, y2 in coords:
        corrected_coords.append((
            x1 + offsets["left"],
            y1 + offsets["top"],
            x2 + offsets["left"],
            y2 + offsets["top"]
        ))
    return corrected_coords

def createAndImgWithBoundingBox(src_img, contours, inverse_bin_img):
    for i in range(len(contours)):
        bounding_box = cv2.boundingRect(contours[i])
        if not judgeAreaOfBoundingBox(bounding_box, src_img.shape[0]*src_img.shape[1]):
            continue
        # Draw rectangle
        cv2.rectangle(src_img, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (255), 3)

    dst_img = cv2.bitwise_and(src_img, inverse_bin_img)
    return dst_img

def judgeAreaOfBoundingBox(bounding_box, page_area):
    bb_area = bounding_box[2] * bounding_box[3]
    if bb_area < 0.048 * page_area:
        return False
    return True

def judgeBoundingBoxOverlap(isOverlap,bounding_boxes, brect):
    for box in bounding_boxes:
    # If it's the same, skip
        if box[0] == brect[0] and box[1] == brect[1] and box[2] == brect[2] and box[3] == brect[3]:
            continue

        overlap_rect = cv2.bitwise_and(brect, box)
        if overlap_rect[0] == 0 and overlap_rect[1] == 0 and overlap_rect[2] == 0 and overlap_rect[3] == 0:
            continue

        if (overlap_rect[0] == brect[0] and overlap_rect[1] == brect[1]
            and overlap_rect[2] == brect[2] and overlap_rect[3] == brect[3]):
            isOverlap = False

    return isOverlap

import numpy as np

def definePanelCorners(definite, current_point, bounding_box_min_dist, PageCornerPoint, definite_panel_point, boundingBoxPoint):
    if not definite:
        page_corner_dist = np.linalg.norm(np.array(boundingBoxPoint) - np.array(PageCornerPoint))
        if page_corner_dist < 8:
            definite_panel_point = PageCornerPoint
            definite = True
        else:
            bounding_box_dist = np.linalg.norm(np.array(boundingBoxPoint) - np.array(current_point))
            if bounding_box_dist < bounding_box_min_dist:
                bounding_box_min_dist = bounding_box_dist
                definite_panel_point = current_point
    # print("受け渡す前",definite_panel_point)
    return definite, bounding_box_min_dist, definite_panel_point

def align2edge(definite_panel_point, inverse_bin_img):
    th_edge = 6  # If within 6px
    for i in range(4):
        x, y = definite_panel_point[i]
        if i in [0, 2] and x < th_edge:  # lt and lb
            x = 0
        if i in [0, 1] and y < th_edge:  # lt and rt
            y = 0
        if i in [1, 3] and x > inverse_bin_img.shape[1] - th_edge:  # rt and rb
            x = inverse_bin_img.shape[1]
        if i in [2, 3] and y > inverse_bin_img.shape[0] - th_edge:  # lb and rb
            y = inverse_bin_img.shape[0]
        definite_panel_point[i] = (x, y)

    return definite_panel_point

def createAlphaImage(alph_img, definitePanelPoint):
    height, width = alph_img.shape
    for i in range(height):
        for j in range(width):
            px = alph_img[i, j]
            # 領域外を指定
            # if outside(definitePanelPoint, renew_line(definitePanelPoint), a, b):
            #     alph_img[i, j] = 0


def renew_line(p):
    top_line = [p[0], p[2],False]
    bottom_line = [p[1], p[3],False]
    left_line = [p[1], p[0],True]
    right_line = [p[3], p[2],True]
    return top_line, bottom_line, left_line, right_line

def calc(line):
    if line[2]:
        a = float(line[1][0] - line[0][0]) / (line[1][1] - line[0][1])
        b = line[0][0] - a * line[0][1]
    else:
        a = float(line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        b = line[0][1] - a * line[0][0]
    return a,b


def outside(p,lines, a,b):
    return judgeArea(p,lines[0][2],a,b) == 1 or judgeArea(p,lines[1][2],a,b) == 0 or judgeArea(p,lines[2][2],a,b) == 1 or judgeArea(p,lines[3][2],a,b) == 0

def judgeArea(p,y2x,a,b):
    if y2x: p = [p[1],p[0]]

    if p[1] > a * p[0] + b:
        return 0
    elif p[1] < a * p[0] + b:
        return 1
    else:
        return 2






if __name__ == '__main__':
    test_img = cv2.imread('./image/comic_dataset/images/ARMS_025.jpg')
    r_img = cut_page(test_img)[0]
    frame_detect(r_img)
