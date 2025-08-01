import cv2
import numpy as np
import math
import os
import re


def PageCut(input_img):
    pageImg = []
    if input_img.shape[1] > input_img.shape[0]:  # 縦 < 横の場合: 見開きだと判断し真ん中で切断
        cut_img_left = input_img[:, : input_img.shape[1] // 2]  # 右ページ
        cut_img_right = input_img[:, input_img.shape[1] // 2 :]  # 左ページ
        pageImg.append(cut_img_right)
        pageImg.append(cut_img_left)
    else:  # 縦 > 横の場合: 単一ページ画像だと判断しそのまま保存
        pageImg.append(input_img)
    return pageImg


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


def is_all_black_image(img, threshold=0):
    if len(img.shape) == 3:  # もしカラー画像なら、グレースケールに変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return np.all(avg_color <= threshold)


def get_imgs_from_folder_sorted(folder):
    image_files = []
    all_files = os.listdir(folder)
    for file in all_files:
        if os.path.isfile(os.path.join(folder, file)):
            extension = os.path.splitext(file)[1].lower()
            if extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                image_files.append(file)
    return sorted(image_files, key=natural_keys)


def get_imgList_form_dir(dir_path):
    img_list = []
    for file in os.listdir(dir_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_list.append(os.path.join(dir_path, file))
    return img_list


def extractSpeechBalloon(fukidashi_contours, hierarchy2, gaussian_img):
    for i in range(len(fukidashi_contours)):
        area = cv2.contourArea(fukidashi_contours[i])
        length = cv2.arcLength(fukidashi_contours[i], True)
        en = 0.0
        if (
            gaussian_img.shape[0] * gaussian_img.shape[1] * 0.005 <= area
            and area < gaussian_img.shape[0] * gaussian_img.shape[1] * 0.05
        ):
            en = 4.0 * np.pi * area / (length * length)
        if en > 0.4:
            cv2.drawContours(gaussian_img, fukidashi_contours, i, 0, -1, cv2.LINE_AA, hierarchy2, 1)


def findFrameArea(input_page_image):
    # 入力がグレースケールでない場合はグレースケールに変換
    if len(input_page_image.shape) > 2:
        input_page_image = cv2.cvtColor(input_page_image, cv2.COLOR_BGR2GRAY)
    # ガウシアンフィルタ
    gaussian_img = np.zeros(input_page_image.shape, dtype=int)
    gaussian_img = cv2.GaussianBlur(input_page_image, (3, 3), 0)  # Smoothing

    # Generation of inverse binarized image
    inverse_bin_img = cv2.bitwise_not(gaussian_img)  # Inverse image
    _, inverse_bin_img = cv2.threshold(inverse_bin_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binarization

    # Histogram generation for left and right
    histgram_lr = np.zeros(inverse_bin_img.shape[1], dtype=int)
    for y in range(inverse_bin_img.shape[0]):
        for x in range(inverse_bin_img.shape[1]):
            if x <= 2 or x >= inverse_bin_img.shape[1] - 2 or y <= 2 or y >= inverse_bin_img.shape[0] - 2:
                continue
            if inverse_bin_img[y, x] > 0:
                histgram_lr[x] += 1

    min_x_lr = np.argmax(histgram_lr > 0)
    max_x_lr = inverse_bin_img.shape[1] - np.argmax(histgram_lr[::-1] > 0)

    # Erroneous values are pushed towards the edges
    min_x_lr = 0 if min_x_lr < 6 else min_x_lr
    max_x_lr = inverse_bin_img.shape[1] if max_x_lr > inverse_bin_img.shape[1] - 6 else max_x_lr

    cut_page_img_lr = input_page_image[:, min_x_lr:max_x_lr]

    # Histogram generation for top and bottom
    histgram_tb = np.zeros(inverse_bin_img.shape[0], dtype=int)
    for y in range(inverse_bin_img.shape[0]):
        for x in range(inverse_bin_img.shape[1]):
            if x <= 2 or x >= inverse_bin_img.shape[1] - 2 or y <= 2 or y >= inverse_bin_img.shape[0] - 2:
                continue
            if inverse_bin_img[y, x] > 0:
                histgram_tb[y] += 1

    min_y_tb = np.argmax(histgram_tb > 0)
    max_y_tb = inverse_bin_img.shape[0] - np.argmax(histgram_tb[::-1] > 0)

    # Erroneous values are pushed towards the edges
    min_y_tb = 0 if min_y_tb < 6 else min_y_tb
    max_y_tb = cut_page_img_lr.shape[0] if max_y_tb > cut_page_img_lr.shape[0] - 6 else max_y_tb

    cut_page_img = cut_page_img_lr[min_y_tb:max_y_tb, :]

    return cut_page_img


# 0-white 1-black
def get_page_type(src_image):
    BLACK_LENGTH_TH = 4
    input_page_image = src_image.copy()
    frame_exist_page = findFrameArea(input_page_image)
    _, frame_exist_page = cv2.threshold(frame_exist_page, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # top
    page_type = 1
    for y in range(3, BLACK_LENGTH_TH):
        for x in range(frame_exist_page.shape[1]):
            if frame_exist_page[y, x] != 0:
                page_type = 0
                break
        if not page_type:
            break
    if page_type:
        return page_type

    # bottom
    page_type = 1
    for y in range(frame_exist_page.shape[0] - 3, frame_exist_page.shape[0] - BLACK_LENGTH_TH, -1):
        for x in range(frame_exist_page.shape[1]):
            if frame_exist_page[y, x] != 0:
                page_type = 0
                break
        if not page_type:
            break
    if page_type:
        return page_type

    # right
    page_type = 1
    for y in range(frame_exist_page.shape[0]):
        for x in range(
            frame_exist_page.shape[1] - 3,
            frame_exist_page.shape[1] - BLACK_LENGTH_TH,
            -1,
        ):
            if frame_exist_page[y, x] != 0:
                page_type = 0
                break
        if not page_type:
            break
    if page_type:
        return page_type

    # left
    page_type = 1
    for y in range(frame_exist_page.shape[0]):
        for x in range(3, BLACK_LENGTH_TH):
            if frame_exist_page[y, x] != 0:
                page_type = 0
                break
        if not page_type:
            break
    if page_type:
        return page_type

    return page_type


class Line:
    def __init__(self, p1=np.array([0, 0]), p2=np.array([0, 0]), y2x=False):
        self.y2x = y2x
        self.p1 = p1
        self.p2 = p2
        self.a, self.b = 0, 0
        self.calc()

    def calc(self):
        if self.y2x:
            self.p1 = self.p1[::-1]
            self.p2 = self.p2[::-1]

        if self.p2[0] - self.p1[0] != 0:
            self.a = (self.p2[1] - self.p1[1]) / (self.p2[0] - self.p1[0])
            self.b = self.p1[1] - self.a * self.p1[0]

    def judgeArea(self, p):
        if self.y2x:
            p = p[::-1]

        if p[1] < (self.a * p[0] + self.b):
            return 1
        else:
            return 0


class Points:
    def __init__(
        self,
        lt=np.array([0, 0]),
        lb=np.array([0, 0]),
        rt=np.array([0, 0]),
        rb=np.array([0, 0]),
    ):
        self.lt = lt if lt is not None else cv2.Point(0, 0)
        self.lb = lb if lb is not None else cv2.Point(0, 0)
        self.rt = rt if rt is not None else cv2.Point(0, 0)
        self.rb = rb if rb is not None else cv2.Point(0, 0)
        self.renew_line()

    def renew_line(self):
        self.top_line = Line(self.lt, self.rt)
        self.bottom_line = Line(self.lb, self.rb)
        self.left_line = Line(self.lt, self.lb, True)
        self.right_line = Line(self.rt, self.rb, True)

    def outside(self, p):
        return (
            self.top_line.judgeArea(p) == 1
            or self.right_line.judgeArea(p) == 0
            or self.bottom_line.judgeArea(p) == 0
            or self.left_line.judgeArea(p) == 1
        )


class Framedetect:
    def __init__(self):
        self.pageCorners = []  # リストで初期化

    def frame_detect(self, input_img):
        img_size = input_img.shape
        pageCorners = Points()
        color_page = input_img.copy()
        if len(color_page.shape) == 2 or color_page.shape[2] == 1:
            color_page = cv2.cvtColor(color_page, cv2.COLOR_GRAY2BGR)
        zeros = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        # 入力画像がカラーの場合
        if len(img_size) == 3:
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = input_img

        binForSpeechBalloon_img = None
        panel_images = []

        _, binForSpeechBalloon_img = cv2.threshold(gray_img.copy(), 230, 255, cv2.THRESH_BINARY)  # Binarization

        binForSpeechBalloon_img = cv2.erode(binForSpeechBalloon_img, None, iterations=1)  # Erosion
        binForSpeechBalloon_img = cv2.dilate(binForSpeechBalloon_img, None, iterations=1)  # Dilation

        hierarchy2 = []
        hukidashi_contours = []  # 輪郭情報格納用リスト

        # 吹き出し塗りつぶしのための輪郭抽出
        hukidashi_contours, hierarchy2 = cv2.findContours(binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # 平滑化
        gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        # 吹き出しによるコマ未検出を防ぐため吹き出し検出で塗りつぶし
        self.extractSpeechBalloon(hukidashi_contours, hierarchy2, gaussian_img)

        # 階調反転二値画像
        inverse_bin_img = np.zeros(img_size, dtype=np.uint8)
        inverse_bin_img = cv2.threshold(gaussian_img, 210, 255, cv2.THRESH_BINARY_INV)[1]

        # コマ存在領域推定
        self.findFrameExistenceArea(inverse_bin_img)

        # キャニーフィルタ画像
        canny_img = cv2.Canny(gray_img, 120, 130, 3)

        lines = cv2.HoughLines(canny_img, 1, np.pi / 180.0, 50)  # Detect lines
        lines2 = cv2.HoughLines(canny_img, 1, np.pi / 360.0, 50)

        lines_img = np.zeros(img_size, dtype=np.uint8)
        print("lines_img.shape", lines_img.shape)

        self.drawHoughLines(lines, lines_img)
        self.drawHoughLines2(lines2, lines_img)

        # 論理積画像の作成
        and_img = np.zeros(img_size, dtype=np.uint8)
        # 二値画像と直線検出画像の論理積の計算

        and_img = cv2.bitwise_and(inverse_bin_img, cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY))

        # 輪郭検出（ラベリング）
        contours, _ = cv2.findContours(and_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundingbox_from_and_img = and_img.copy()

        complement_and_img = np.zeros(img_size, dtype=np.uint8)
        self.createAndImgWithBoundingBox(boundingbox_from_and_img, contours, inverse_bin_img, complement_and_img)
        print("complement_and_img.shape", complement_and_img.shape)

        contours3 = []
        bounding_boxes = []  # バウンディングボックス群

        # 輪郭検出（ラベリング）
        (
            contours3,
            _,
        ) = cv2.findContours(
            cv2.cvtColor(complement_and_img, cv2.COLOR_BGR2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # バウンディングボックスの登録
        for i in range(len(contours3)):
            # バウンディングボックスの取得
            tmp_boundhging_box = tmp_bounding_box = cv2.boundingRect(contours3[i])

            if self.judge_area_of_bounding_box(
                tmp_bounding_box,
                complement_and_img.shape[0] * complement_and_img.shape[1],
            ):
                # バウンディングボックスの登録
                bounding_boxes.append(tmp_bounding_box)

        # ページでコマの輪郭描画
        for i in range(len(contours3)):
            # 近似の座標
            approx = []

            # 輪郭を直線近似する
            approx = cv2.approxPolyDP(contours3[i], 6, True)

            # 外接矩形の生成
            brect = cv2.boundingRect(contours3[i])

            # 左下，右上の座標(xmin,ymin)(xmax,ymax)
            xmin = brect[0]
            ymin = brect[1]
            xmax = brect[0] + brect[2]
            ymax = brect[1] + brect[3]

            # 端に寄せる
            if xmin < 6:
                xmin = 0
            if xmax > inverse_bin_img.shape[1] - 6:
                xmax = inverse_bin_img.shape[1]
            if ymin < 6:
                ymin = 0
            if ymax > inverse_bin_img.shape[0] - 6:
                ymax = inverse_bin_img.shape[0]

            # バウンディングボックス（外接矩形）の4点

            bbPoints = Points(
                cv2.Point(xmin, ymin),
                cv2.Point(xmin, ymax),
                cv2.Point(xmax, ymin),
                cv2.Point(xmax, ymax),
            )

            # 最終的な代表点
            definitePanelPoint = Points()

            # 点が確定か否か
            flag_lt = False
            flag_lb = False
            flag_rt = False
            flag_rb = False

            # 最小距離初期値（ページの高さにしておくことで必ず更新される）
            bb_min_lt = input_img.shape[0]
            bb_min_lb = input_img.shape[0]
            bb_min_rt = input_img.shape[0]
            bb_min_rb = input_img.shape[0]

            isOverlap = True

            if self.judgeAreaOfBoundingBox(brect, input_img.shape[0] * input_img.shape[1]):
                self.judgeBoundingBoxOverlap(bounding_boxes, brect, isOverlap)
            else:
                isOverlap = False

            if not isOverlap:
                continue

            for i in range(len(approx)):
                p = cv2.Point(approx[i][0][0], approx[i][0][1])

                self.definePanelCorners(
                    flag_lt,
                    p,
                    bb_min_lt,
                    self.pageCorners.lt,
                    definitePanelPoint.lt,
                    bbPoints.lt,
                )
                self.definePanelCorners(
                    flag_lb,
                    p,
                    bb_min_lb,
                    self.pageCorners.lb,
                    definitePanelPoint.lb,
                    bbPoints.lb,
                )
                self.definePanelCorners(
                    flag_rt,
                    p,
                    bb_min_rt,
                    self.pageCorners.rt,
                    definitePanelPoint.rt,
                    bbPoints.rt,
                )
                self.definePanelCorners(
                    flag_rb,
                    p,
                    bb_min_rb,
                    self.pageCorners.rb,
                    definitePanelPoint.rb,
                    bbPoints.rb,
                )
                self.align2edge(definitePanelPoint, inverse_bin_img)

                definitePanelPoint.renew_line()

                # 透過画像（４ちゃんねる）
                alpha_img = np.zeros((input_img.shape[0], input_img.shape[1], 4), dtype=np.uint8)

                # rgb->rgba
                alpha_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2RGBA)

                # コマ以外を透過
                self.createAlphaImage(alpha_img, definitePanelPoint)

                # サイズを合わせる
                cut_img = alpha_img[brect.y : brect.y + brect.height, brect.x : brect.x + brect.width]

                # 保存
                panel_images.append(cut_img)

                # コマ線を描画
                cv2.line(
                    color_page,
                    definitePanelPoint.lt,
                    definitePanelPoint.rt,
                    (255, 0, 0),
                    2,
                )
                cv2.line(
                    color_page,
                    definitePanelPoint.rt,
                    definitePanelPoint.rb,
                    (255, 0, 0),
                    2,
                )
                cv2.line(
                    color_page,
                    definitePanelPoint.rb,
                    definitePanelPoint.lb,
                    (255, 0, 0),
                    2,
                )
                cv2.line(
                    color_page,
                    definitePanelPoint.lb,
                    definitePanelPoint.lt,
                    (255, 0, 0),
                    2,
                )

        return panel_images

    def extractSpeechBalloon(self, fukidashi_contours, hierarchy2, gaussian_img):
        for i in range(len(fukidashi_contours)):
            area = cv2.contourArea(fukidashi_contours[i])
            length = cv2.arcLength(fukidashi_contours[i], True)
            en = 0

            if (
                gaussian_img.shape[0] * gaussian_img.shape[1] * 0.008 <= area
                and area < gaussian_img.shape[0] * gaussian_img.shape[1] * 0.03
            ):
                en = 4.0 * math.pi * area / (length * length)
                if en > 0.4:
                    cv2.drawContours(
                        gaussian_img,
                        fukidashi_contours,
                        i,
                        0,
                        -1,
                        cv2.LINE_AA,
                        hierarchy2,
                        1,
                    )

    def findFrameExistenceArea(self, inverse_bin_img):
        histogram = np.zeros(inverse_bin_img.shape[1], dtype=np.int)
        pageCorners = Points()

        for y in range(inverse_bin_img.shape[0]):
            for x in range(inverse_bin_img.shape[1]):
                if x <= 2 or x >= inverse_bin_img.shape[1] - 2 or y <= 2 or y >= inverse_bin_img.shape[0] - 2:
                    continue
                if 0 < inverse_bin_img[y, x]:
                    histogram[x] += 1

        min_x = 0
        max_x = inverse_bin_img.shape[1] - 1

        for x in range(inverse_bin_img.shape[1]):
            if 0 < histogram[x]:
                min_x = x
                break

        for x in range(inverse_bin_img.shape[1] - 1, -1, -1):
            if 0 < histogram[x]:
                max_x = x
                break

        if min_x < 6:
            min_x = 0
        if max_x > inverse_bin_img.shape[1] - 6:
            max_x = inverse_bin_img.shape[1]

        pageCorners.lt = (min_x, 0)
        pageCorners.rt = (max_x, 0)
        pageCorners.lb = (min_x, inverse_bin_img.shape[0])
        pageCorners.rb = (max_x, inverse_bin_img.shape[0])

        rec_img = np.zeros((inverse_bin_img.shape[0], inverse_bin_img.shape[1], 3), np.uint8)

    def drawHoughLines(self, lines, drawLinesImage):
        for i in range(min(len(lines), 100)):
            line = lines[i]

            rho = line[0][0]  # ρ
            theta = line[0][0]  # θ

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            pt1 = np.zeros((2,), dtype=np.float32)
            pt2 = np.zeros((2,), dtype=np.float32)

            pt1[0] = x0 - 2000 * b
            pt1[1] = y0 + 2000 * a
            pt2[0] = x0 + 2000 * b
            pt2[1] = y0 - 2000 * a

            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            cv2.line(drawLinesImage, pt1, pt2, (255), 1, cv2.LINE_AA)

    def drawHoughLines2(self, lines, drawLinesImage):
        for i in range(min(len(lines), 100)):
            line = lines[i]

            rho = line[0][0]  # ρ
            theta = line[0][0]  # θ

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            pt1 = np.zeros((2,), dtype=np.float32)
            pt2 = np.zeros((2,), dtype=np.float32)

            pt1[0] = x0 - 2000 * b
            pt1[1] = y0 + 2000 * a
            pt2[0] = x0 + 2000 * b
            pt2[1] = y0 - 2000 * a

            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            cv2.line(drawLinesImage, pt1, pt2, (255), 1, cv2.LINE_AA)

    def createAndImgWithBoundingBox(self, src_img, contours, inverse_bin_img, dst_img):
        dst_img = np.zeros_like(src_img)
        for contour in contours:
            bounding_box = cv2.boundingRect(contour)
            if not self.judgeAreaOfBoundingBox(bounding_box, src_img.shape[0] * src_img.shape[1]):
                continue
            cv2.rectangle(
                src_img,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                255,
                3,
            )
        cv2.bitwise_and(src_img, inverse_bin_img, dst_img)
        return dst_img

    def judgeAreaOfBoundingBox(self, bounding_box, page_area):
        return bounding_box[2] * bounding_box[3] >= 0.048 * page_area

    def judgeBoundingBoxOverlap(self, isOverlap, bounding_boxes, brect):
        isOverlap = True
        for bounding_box in bounding_boxes:
            if (
                bounding_box[0] == brect[0]
                and bounding_box[1] == brect[1]
                and bounding_box[2] == brect[2]
                and bounding_box[3] == brect[3]
            ):
                continue
            overlap_rect = self._rectIntersection(brect, bounding_box)
            if (
                overlap_rect[0] == brect[0]
                and overlap_rect[1] == brect[1]
                and overlap_rect[2] == brect[2]
                and overlap_rect[3] == brect[3]
            ):
                isOverlap = False
        return isOverlap

    def definePanelCorners(
        self,
        definite,
        currentPoint,
        boundingBoxMinDist,
        PageCornerPoint,
        definitePanelPoint,
        boundingBoxPoint,
    ):
        if not definite:
            pageCornerDist = np.linalg.norm(np.array(boundingBoxPoint) - np.array(PageCornerPoint))
            if pageCornerDist < 8:
                definitePanelPoint = PageCornerPoint
                definite = True
            else:
                boundingBoxDist = np.linalg.norm(np.array(boundingBoxPoint) - np.array(currentPoint))
                if boundingBoxDist < boundingBoxMinDist:
                    boundingBoxMinDist = boundingBoxDist
                    definitePanelPoint = currentPoint
        return definite, boundingBoxMinDist, definitePanelPoint

    def align2edge(self, definitePanelPoint, inverse_bin_img):
        th_edge = 6
        if definitePanelPoint[0] < th_edge:
            definitePanelPoint[0] = 0
        if definitePanelPoint[1] < th_edge:
            definitePanelPoint[1] = 0
        if definitePanelPoint[0] > inverse_bin_img.shape[1] - th_edge:
            definitePanelPoint[0] = inverse_bin_img.shape[1]
        if definitePanelPoint[1] > inverse_bin_img.shape[0] - th_edge:
            definitePanelPoint[1] = inverse_bin_img.shape[0]
        return definitePanelPoint

    def createAlphaImage(self, alphaImage, definitePanelPoint):
        for y in range(alphaImage.shape[0]):
            for x in range(alphaImage.shape[1]):
                px = alphaImage[y, x]
                if self.outside(definitePanelPoint, (x, y)):
                    px[3] = 0  # Set alpha channel to 0
                    alphaImage[y, x] = px
        return alphaImage


def detect_speech_balloons(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    speech_balloons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        if area > 500 and circularity > 0.1:
            x, y, w, h = cv2.boundingRect(contour)
            balloon = img[y : y + h, x : x + w]
            balloon_mask = np.zeros(balloon.shape[:2], np.uint8)
            cv2.drawContours(balloon_mask, [contour], -1, 255, -1)
            balloon[np.where(balloon_mask == 0)] = 0
            speech_balloons.append(balloon)
    return speech_balloons
