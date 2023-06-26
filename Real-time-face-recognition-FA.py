import cv2

# بارگیری فایل Haar cascade برای تشخیص چهره
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# باز کردن دوربین
cap = cv2.VideoCapture(0)

while True:
    # خواندن فریم از دوربین
    ret, frame = cap.read()

    # تبدیل فریم به تصویر سیاه و سفید
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # تشخیص چهره در تصویر سیاه و سفید
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # رسم مستطیل دور چهره‌های تشخیص داده شده
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # نمایش فریم با چهره‌های تشخیص داده شده
    cv2.imshow('تشخیص چهره‌ها', frame)

    # قطع حلقه در صورت فشار دادن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزاد کردن دوربین و بستن پنجره‌ها
cap.release()
cv2.destroyAllWindows()
