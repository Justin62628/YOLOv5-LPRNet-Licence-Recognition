import os
import random
import re
import shutil

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def imread(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
    except:
        return None


class GetCcpdDataset:
    def __init__(self, root: str, dataset_redis_root: str, path_config: dict, dataset_ratio=(0.7, 0.2, 0.1)):
        """
        :param: path_config: {'dir_path': target_len}
        """
        self.root = root
        self.dataset_redis_root = dataset_redis_root
        self.path_config = path_config
        self.img_paths = list()
        self.img_datas = list()

        self.train_data = list()
        self.valid_data = list()
        self.test_data = list()
        self.index_list = list()
        self.dataset_ratio = dataset_ratio
        assert abs(sum(self.dataset_ratio) - 1.0) < 1e-3

        self.base_license_re = re.compile(
            "\d+-\d+_\d+-\d+&\d+_\d+&\d+-(\d+)&(\d+)_(\d+)&(\d+)_(\d+)&(\d+)_(\d+)&(\d+)-(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)-\d+-\d+.*?")
        self.green_license_re = re.compile(
            "\d+-\d+_\d+-\d+&\d+_\d+&\d+-(\d+)&(\d+)_(\d+)&(\d+)_(\d+)&(\d+)_(\d+)&(\d+)-(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)-\d+-\d+.*?")

        self.update_full_datasets()

    @staticmethod
    def get_img_paths(dirpath, target_len):
        img_paths = list()
        for root, _, files in os.walk(dirpath):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext.lower() not in ['.jpg', '.png']:
                    continue
                path = os.path.join(root, file)
                img_paths.append(path)
        random.shuffle(img_paths)
        img_paths = img_paths[:target_len]
        return img_paths

    def update_full_datasets(self):
        # update img path
        for dirpath in self.path_config:
            target_len = self.path_config[dirpath]
            img_paths = self.get_img_paths(os.path.join(self.root, dirpath), target_len)
            self.img_paths.extend(img_paths)

        random.shuffle(self.img_paths)

        # update img data
        for img_path in self.img_paths:
            data = self.get_ccpd_img_data(img_path)
            if data is None:
                continue
            self.img_datas.append([img_path, data])

    def get_ccpd_img_data(self, img_path: str):
        """
        base:  025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        green: 04- 90_267-158&448_542&553-541&553_162&551_158&448_542&450-0_1_3_ 24_27_33_30_24-99-116.jpg
        """
        img = imread(img_path)
        if img is None:
            return
        img_basename = os.path.basename(img_path)
        data_list = self.base_license_re.findall(img_basename)
        is_green = 0
        if not len(data_list):
            data_list = self.green_license_re.findall(img_basename)
            is_green = 1
        if not len(data_list):
            return None
        data_list = data_list[0]
        rbx = int(data_list[0])
        rby = int(data_list[1])
        lbx = int(data_list[2])
        lby = int(data_list[3])
        ltx = int(data_list[4])
        lty = int(data_list[5])
        rtx = int(data_list[6])
        rty = int(data_list[7])

        width = abs(int(rbx) - int(ltx))
        height = abs(int(rby) - int(lty))  # bounding box的宽和高
        cx = float(ltx) + width / 2
        cy = float(lty) + height / 2  # bounding box中心点

        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        # Recognition
        result = ""
        list_plate = data_list[8:]
        result += provinces[int(list_plate[0])]
        result += alphabets[int(list_plate[1])]
        result += ads[int(list_plate[2])] + ads[int(list_plate[3])] + ads[int(list_plate[4])] + ads[
            int(list_plate[5])] + ads[int(list_plate[6])]
        if is_green > 0:
            result += ads[int(list_plate[7])]
            if result[2] not in ['D', 'F'] and result[-1] not in ['D', 'F']:
                print(img_basename)
                print("Error Greenlabel, Please check!")
                return
            # assert 0, "Error label ^~^!!!"
        return is_green, result, ltx, lty, rbx, rby, cx, cy, width, height
        pass

    @staticmethod
    def parse_yolov5_data(img_data: list):
        img_path, data = img_data
        is_green, result, ltx, lty, rbx, rby, cx, cy, width, height = data
        return img_path, [is_green, cx, cy, width, height]

    @staticmethod
    def parse_lpr_data(img_data: list):
        img_path, data = img_data
        is_green, result, ltx, lty, rbx, rby, cx, cy, width, height = data
        return img_path, [is_green, result, ltx, lty, rbx, rby]

    def parse_dataset(self, method):
        train_ratio, valid_ratio, test_ratio = self.dataset_ratio
        dataset_len = len(self.img_datas)
        self.train_data = list(map(method, self.img_datas[:int(dataset_len * train_ratio)]))
        self.valid_data = list(map(method, self.img_datas[int(dataset_len * train_ratio) + 1: int(
            dataset_len * (train_ratio + valid_ratio))]))
        self.test_data = list(map(method, self.img_datas[-int(dataset_len * test_ratio) + 1:]))

    @staticmethod
    def write_dataset_label(dataset_head: str, dataset: list):
        with open(f'{dataset_head}_labels.txt', 'w', encoding='utf-8') as w:
            for d in dataset:
                path, data = d
                w.write(f'"{path}",{",".join([str(element) for element in data])}\n')
        pass

    def write_datasets_label(self, dataset_head: str):
        self.write_dataset_label(f"{dataset_head}_train", self.train_data)
        self.write_dataset_label(f"{dataset_head}_valid", self.valid_data)
        self.write_dataset_label(f"{dataset_head}_test", self.test_data)
        pass

    def transfer(self, parse_method, transfer_method):
        self.parse_dataset(parse_method)
        train_imgs_dir = os.path.join(self.dataset_redis_root, "images", "train")
        val_imgs_dir = os.path.join(self.dataset_redis_root, "images", "val")
        test_imgs_dir = os.path.join(self.dataset_redis_root, "images", "test")
        train_labels_dir = os.path.join(self.dataset_redis_root, "labels", "train")
        val_labels_dir = os.path.join(self.dataset_redis_root, "labels", "val")
        test_labels_dir = os.path.join(self.dataset_redis_root, "labels", "test")
        for d in [train_imgs_dir, val_imgs_dir, test_imgs_dir, train_labels_dir,
                  val_labels_dir, test_labels_dir]:
            os.makedirs(d, exist_ok=True)
        for dataset, img_dir, label_dir in [(self.train_data, train_imgs_dir, train_labels_dir),
                                            (self.valid_data, val_imgs_dir, val_labels_dir),
                                            (self.test_data, test_imgs_dir, test_labels_dir)]:
            for d in tqdm(dataset, total=len(dataset), desc=os.path.basename(img_dir)):
                transfer_method(d, img_dir, label_dir)
        pass

    def transfer_yolov5_method(self, d, img_dir, label_dir):
        """
        Direct Call after update_img_datasets
        """
        path, data = d
        basename = os.path.basename(path)
        shutil.copy(path, os.path.join(img_dir, basename))
        with open(os.path.join(label_dir, os.path.splitext(basename)[0] + '.txt'), 'w', encoding='utf-8') as f:
            f.write(str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + " " + str(data[3]) + " " + str(
                data[4]))

    def transfer_lpr_method(self, d, img_dir, label_dir):
        """
        Direct Call after update_img_datasets
        """
        path, data = d
        # basename = os.path.basename(path)

        _, result, ltx, lty, rbx, rby = data

        img = Image.fromarray(imread(path))
        img = img.crop((ltx, lty, rbx, rby))  # 裁剪出车牌位置
        img = img.resize((94, 24), Image.LANCZOS)
        img = np.asarray(img)  # 转成array,变成24*94*3

        cv2.imencode('.jpg', img)[1].tofile(os.path.join(img_dir, result+'.jpg'))

    def write_all_labels(self):
        """
        Direct Call after update_img_datasets
        """
        # yolov5
        self.parse_dataset(self.parse_yolov5_data)
        self.write_datasets_label('yolov5')
        # lpr
        self.parse_dataset(self.parse_lpr_data)
        self.write_datasets_label('yolov5')


if __name__ == "__main__":
    _path_config = {
        'ccpd_base': int(15.69e3),
        'ccpd_challenge': int(5.034e3),
        'ccpd_db': int(2.069e3),
        'ccpd_fn': int(2.069e3),
        'ccpd_rotate': int(1.069e3),
        'ccpd_tilt': int(1.034e3),
        'ccpd_weather': int(3.034e3),
        'ccpd_green': int(10e3),
    }
    # _gcd = GetCcpdDataset(r"U:\ML\TmpDatasets\CCPD", r"U:\ML\TmpDatasets\CCPD\OwnDet", _path_config)
    # _gcd.write_all_labels()
    # _gcd.transfer(_gcd.parse_yolov5_data, _gcd.transfer_yolov5_method)
    _gcd = GetCcpdDataset(r"U:\ML\TmpDatasets\CCPD", r"U:\ML\TmpDatasets\CCPD\OwnLpr", _path_config)
    _gcd.transfer(_gcd.parse_lpr_data, _gcd.transfer_lpr_method)

