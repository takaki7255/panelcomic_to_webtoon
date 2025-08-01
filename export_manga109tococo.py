from get_frame_bbox import get_framebbox
import os
import cv2
import json

def main():
    manga109_root = "./../Manga109_released_2023_12_07/"
    ano_dir = "./../Manga109_released_2023_12_07/annotations/"
    img_dir = "./../Manga109_released_2023_12_07/images/"
    titles = "./../Manga109_released_2023_12_07/books.txt"
    with open(titles, 'r') as f:
        titles = f.readlines()
    titles = [title.strip() for title in titles]
    # print(titles)
    img_folder_list = os.listdir(img_dir)
    panel_list = []
    panel_list = get_framebbox(ano_dir + titles[0] + '.xml')
    print(panel_list)
    img_num = panel_list.keys()
    # ３桁の数字に変換
    img_num = [str(num).zfill(3) for num in img_num]
    print(img_num)

if __name__ == "__main__":
    main()