import cv2

from get_right_left_frame_list import get_right_left_frame_list
from cut_page import cut_page
from get_frame_bbox import get_framebbox
from panel_order_estimater import panel_order_estimater

def main():
    ano_file = './../Manga109_released_2023_12_07/annotations/ARMS.xml'
    img_folder = './../Manga109_released_2023_12_07/images/ARMS/'
    panel_list = get_framebbox(ano_file)
    # print(panel_list)
    for page_index, frame_list in panel_list.items():
        #page_indexを３桁の数字に変換
        page_index = str(page_index).zfill(3)
        img = cv2.imread(img_folder + page_index + '.jpg')
        imgs = cut_page(img)
        anos = get_right_left_frame_list(frame_list, img.shape[1])
        for img, frame_list in zip(imgs, anos):
            img_width = img.shape[1]
            img_height = img.shape[0]
            print(frame_list)
            ordered_frame_list = panel_order_estimater(frame_list, img_width, img_height)
            for frame in ordered_frame_list:
                x1 = frame['xmin']
                y1 = frame['ymin']
                x2 = frame['xmax']
                y2 = frame['ymax']
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == '__main__':
    main()