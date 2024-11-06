import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from sklearn.datasets import load_iris

# Hàm tải ảnh từ thư mục
def tai_anh_nha_khoa_tu_thu_muc(duong_dan_thu_muc):
    anh = []
    nhan = []
    for ten_tap_tin in os.listdir(duong_dan_thu_muc):
        nhan_anh = ten_tap_tin.split('_')[0]  # Giả sử nhãn là phần đầu tiên của tên file
        duong_dan_anh = os.path.join(duong_dan_thu_muc, ten_tap_tin)
        img = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)).flatten()  # Resize và chuyển thành vector 1D
            anh.append(img)
            nhan.append(nhan_anh)
    return np.array(anh), np.array(nhan)

# Hàm tải ảnh từ tệp TFRecord
def tai_anh_nha_khoa_tu_tfrecord(duong_dan_tfrecord):
    du_lieu_raw = tf.data.TFRecordDataset(duong_dan_tfrecord)

    def phan_tich_vi_du(vi_du_proto):
        mo_ta_dac_trung = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        vi_du = tf.io.parse_single_example(vi_du_proto, mo_ta_dac_trung)
        anh = tf.image.decode_jpeg(vi_du['image_raw'])  # Giả sử ảnh là JPEG
        nhan = vi_du['label']
        anh = tf.image.resize(anh, [128, 128])
        anh = tf.image.rgb_to_grayscale(anh)
        return anh, nhan

    du_lieu_phan_tich = du_lieu_raw.map(phan_tich_vi_du)
    anh, nhan = [], []
    for img, nhan_anh in du_lieu_phan_tich:
        anh.append(img.numpy().flatten())
        nhan.append(nhan_anh.numpy())
    return np.array(anh), np.array(nhan)

# Hàm huấn luyện và đánh giá mô hình
def huan_luyen_va_danh_gia(X, y, tieu_chi="gini"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    mo_hinh = DecisionTreeClassifier(criterion=tieu_chi, random_state=42)
    mo_hinh.fit(X_train, y_train)
    
    y_du_doan = mo_hinh.predict(X_test)
    do_chinh_xac = accuracy_score(y_test, y_du_doan)
    bao_cao = classification_report(y_test, y_du_doan)
    
    print(f"Mô hình: Decision Tree ({tieu_chi})")
    print(f"Độ chính xác: {do_chinh_xac:.2f}")
    print("Báo cáo phân loại:")
    print(bao_cao)

# Chọn dữ liệu và chạy huấn luyện
def main():
    # Dùng IRIS dataset làm mẫu
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    print("Sử dụng bộ dữ liệu IRIS:")
    huan_luyen_va_danh_gia(X_iris, y_iris, tieu_chi="gini")  # CART
    huan_luyen_va_danh_gia(X_iris, y_iris, tieu_chi="entropy")  # ID3

    # Đọc ảnh từ thư mục hoặc từ tệp TFRecord
    # Thay thế đường dẫn bằng đường dẫn phù hợp
    duong_dan_thu_muc = r'C:\Users\tieub\OneDrive\Desktop\PyThon\bai122\Dental_Xray3.tfrec'
    duong_dan_tfrecord = r'C:\Users\tieub\OneDrive\Desktop\PyThon\bai122\Dental_Xray3.tfrec'
    
    # Nếu bạn có thư mục chứa ảnh, hãy dùng đoạn mã dưới:
    X_nha_khoa, y_nha_khoa = tai_anh_nha_khoa_tu_thu_muc(duong_dan_thu_muc)
    
    # Nếu bạn có tệp TFRecord, hãy dùng đoạn mã dưới:
    X_nha_khoa, y_nha_khoa = tai_anh_nha_khoa_tu_tfrecord(duong_dan_tfrecord)

    # Huấn luyện trên dữ liệu ảnh nha khoa (sau khi đã tải dữ liệu vào X_nha_khoa, y_nha_khoa)
    print("Sử dụng bộ dữ liệu nha khoa:")
    huan_luyen_va_danh_gia(X_nha_khoa, y_nha_khoa, tieu_chi="gini")  # CART
    huan_luyen_va_danh_gia(X_nha_khoa, y_nha_khoa, tieu_chi="entropy")  # ID3

# Chạy hàm main
if __name__ == "__main__":
    main()
