# Model Training 
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Generator untuk training dan validasi
train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Direktori dataset 
train_generator = train_data_gen.flow_from_directory(r'C:\Users\Asus\Downloads\archive\train', target_size=(48, 48), batch_size=32, color_mode='grayscale', class_mode='categorical')
validation_generator = validation_data_gen.flow_from_directory(r'C:\Users\Asus\Downloads\archive\test', target_size=(48, 48), batch_size=32, color_mode='grayscale', class_mode='categorical')

# Mode
model = Sequential()

# Convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # Assuming 7 classes for 7 expressions

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_generator, epochs=50, validation_data=validation_generator)

# Save model
model.save('facial_expression_model.h5')

# Real-time facial expression detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = roi_gray / 255.0
        
        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        emotion = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')[max_index]
        
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Facial Expression Detection', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
