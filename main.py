import cv2
import os
from cut_page import cut_page
from get_right_left_frame_list import get_right_left_frame_list
from detect_panel import detect_panel
from panel_order_estimater import panel_order_estimater
from translate_layout import translate_layout
from translate_layout_noresize import translate_layout_center
def main():
    tests_dir = './test/'
    img_list = os.listdir(tests_dir)
    # img_path = './image/comic_dataset/images/ARMS_014.jpg'
    # file_name = img_path.split('/')[-1]
    # file_name = file_name.split('.')[0]
    for img_path in img_list:
        img_path = tests_dir + img_path
        print(img_path)
        file_name = img_path.split('/')[-1]
        file_name = file_name.split('.')[0]
        img = cv2.imread(img_path)
        if img is None:
            print(f'{img_path}No image found.')
        img_width = img.shape[1]
        #コマの検出
        panel_list = detect_panel(img) #検出したコマ座標を返す
        
        r_l_panels = get_right_left_frame_list(panel_list, img_width) #左右のコマ座標を返す
        r_l_imgs = cut_page(img) #左右の画像を返す
        r_l = ['right', 'left']
        for img, frame_list, r_l in zip(r_l_imgs, r_l_panels, r_l):
            img_width = img.shape[1]
            img_height = img.shape[0]
            #コマの順序推定
            ordered_frame_list = panel_order_estimater(frame_list, img_width, img_height)
            
            #レイアウト変換
            # translated_img = translate_layout(img, ordered_frame_list) #リサイズあり
            translated_img = translate_layout_center(img, ordered_frame_list) #リサイズなし中央配置
            if translated_img is None or translated_img.shape[0] == 0 or translated_img.shape[1] == 0:
                print(f'{file_name}_{r_l}No translated image found.')
                continue
            # cv2.imshow('translated_img', translated_img)
            cv2.imwrite(f'./translated_imgs//{file_name}_{r_l}_translated.jpg', translated_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()