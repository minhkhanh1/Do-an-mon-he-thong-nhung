import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Tải tệp CSV
df = pd.read_csv(r"C:\Users\ACER\Desktop\traning\age_gender.csv")

# Chuyển đổi dữ liệu pixel thành mảng numpy và thay đổi kích thước
def preprocess_pixels(pixel_string, image_size=(48, 48)):
    pixels = np.array(pixel_string.split(), dtype='float32')
    image = pixels.reshape(image_size[0], image_size[1])
    image = image / 255.0  # Chuẩn hóa giá trị pixel
    return image

# Tải ảnh và nhãn
X = np.array([preprocess_pixels(pixels) for pixels in df['pixels']])
X = X[..., np.newaxis]  # Thêm kích thước kênh nếu là ảnh grayscale
y_age = df['age'].values
y_gender = df['gender'].values

# Mã hóa nhãn phân loại
gender_encoder = LabelEncoder()
y_gender_encoded = gender_encoder.fit_transform(y_gender)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (90% train, 10% test)
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    X, y_age, y_gender_encoded, test_size=0.15, random_state=42
)

# Định nghĩa kiến trúc mô hình
input_shape = (48, 48, 1)  # Ảnh grayscale
input_layer = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Dự đoán tuổi
age_output = Dense(1, name='age')(x)

# Dự đoán giới tính
gender_output = Dense(2, activation='softmax', name='gender')(x)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])

# Biên dịch mô hình
model.compile(
    optimizer=Adam(),
    loss={'age': 'mean_squared_error', 'gender': 'sparse_categorical_crossentropy'},
    metrics={'age': 'mae', 'gender': 'accuracy'}
)

# Huấn luyện mô hình
history = model.fit(
    X_train, {'age': y_age_train, 'gender': y_gender_train},
    validation_data=(X_test, {'age': y_age_test, 'gender': y_gender_test}),
    epochs=1000,
    batch_size=32
)

# Đường dẫn đến Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "model.h5")

# Lưu mô hình đã huấn luyện
model.save(desktop_path)

# Tải mô hình từ tệp
loaded_model = load_model(desktop_path)

# Kiểm tra mô hình đã tải
print("Tóm tắt mô hình đã tải:")
loaded_model.summary()

# Đánh giá mô hình đã tải
loss, age_loss, gender_loss, age_mae, gender_accuracy = loaded_model.evaluate(
    X_test, {'age': y_age_test, 'gender': y_gender_test}
)
print(f'MAE tuổi: {age_mae:.2f}')
print(f'Độ chính xác giới tính: {gender_accuracy:.2f}')
