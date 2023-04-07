1、先用filter_condition.py过滤固定大小的边界框
2、过滤后的文件转换为yolo格式
3、用create_single_face.py将yolo格式的数据截取小块的人脸
4、用resume_lmk.py将小块的人脸标注转回原图

需要标注的数据地址：D:\Users\yl3146\Desktop\tta\multask\justface\thersh12
过滤数据地址：G:\hp_tracking_proj\three_identy_data\padding
