"""
==============================================
  Face & Eye Detector — Computer Vision Project
  المادة: Computer Vision
  المكتبات: OpenCV, NumPy
==============================================

الاستخدام:
  python detect.py --image صورتك.jpg
  python detect.py --image صورتك.jpg --output output/result.jpg
  python detect.py --camera   ← لفتح الكاميرا مباشرة
"""

import cv2
import numpy as np
import argparse
import os
import sys


# ─────────────────────────────────────────
#  إعداد الـ Haar Cascade classifiers
# ─────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
SMILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)


# ─────────────────────────────────────────
#  الدالة الرئيسية للكشف
# ─────────────────────────────────────────
def detect(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    تأخذ صورة (BGR numpy array) وترجع:
      - الصورة مع الرسومات
      - dict فيه عدد الوجوه والعيون
    """
    output = image.copy()
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- كشف الوجوه ---
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    total_eyes   = 0
    total_smiles = 0
    face_count   = len(faces) if len(faces) > 0 else 0

    for i, (fx, fy, fw, fh) in enumerate(faces):
        # رسم مستطيل الوجه
        cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh),
                      color=(0, 200, 0), thickness=2)

        # ليبل فوق المستطيل
        label = f"Face {i + 1}"
        cv2.rectangle(output, (fx, fy - 28), (fx + 110, fy),
                      (0, 200, 0), -1)
        cv2.putText(output, label, (fx + 5, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # --- كشف العيون داخل منطقة الوجه فقط ---
        roi_gray  = gray[fy:fy + fh, fx:fx + fw]
        roi_color = output[fy:fy + fh, fx:fx + fw]

        eyes = EYE_CASCADE.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(25, 25)
        )

        for (ex, ey, ew, eh) in eyes:
            # مستطيل العين بالأزرق
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                          color=(255, 80, 0), thickness=2)
            # نقطة في المنتصف
            cx, cy = ex + ew // 2, ey + eh // 2
            cv2.circle(roi_color, (cx, cy), 3, (0, 0, 255), -1)
            total_eyes += 1

        # --- كشف الابتسامة في النصف السفلي من الوجه ---
        lower_gray  = roi_gray[fh // 2:, :]
        lower_color = roi_color[fh // 2:, :]

        smiles = SMILE_CASCADE.detectMultiScale(
            lower_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25)
        )

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(lower_color, (sx, sy), (sx + sw, sy + sh),
                          color=(0, 180, 255), thickness=2)
            total_smiles += 1

    # ─── معلومات إجمالية على الصورة ───
    info_lines = [
        f"Faces  : {face_count}",
        f"Eyes   : {total_eyes}",
        f"Smiles : {total_smiles}",
    ]

    panel_h = len(info_lines) * 28 + 16
    cv2.rectangle(output, (8, 8), (200, 8 + panel_h), (30, 30, 30), -1)
    cv2.rectangle(output, (8, 8), (200, 8 + panel_h), (0, 200, 0), 1)

    for j, line in enumerate(info_lines):
        cv2.putText(output, line,
                    (16, 32 + j * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 120), 2)

    stats = {
        "faces": face_count,
        "eyes": total_eyes,
        "smiles": total_smiles,
    }
    return output, stats


# ─────────────────────────────────────────
#  وضع الكاميرا (Real-time)
# ─────────────────────────────────────────
def run_camera():
    print("📷 فتح الكاميرا... اضغط Q للخروج")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ مش قادر يفتح الكاميرا!")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, stats = detect(frame)
        cv2.imshow("Face & Eye Detector — Press Q to quit", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ تم إغلاق الكاميرا")


# ─────────────────────────────────────────
#  وضع الصورة الثابتة
# ─────────────────────────────────────────
def run_image(image_path: str, output_path: str = None):
    if not os.path.exists(image_path):
        print(f"❌ الصورة مش موجودة: {image_path}")
        sys.exit(1)

    print(f"🔍 بيحلل الصورة: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print("❌ مش قادر يقرأ الصورة. تأكد إن الملف صورة صحيحة (jpg/png)")
        sys.exit(1)

    result, stats = detect(image)

    # طباعة النتائج
    print("─" * 35)
    print(f"  ✅ الوجوه    : {stats['faces']}")
    print(f"  👁  العيون   : {stats['eyes']}")
    print(f"  😊 الابتسامات: {stats['smiles']}")
    print("─" * 35)

    # حفظ أو عرض
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"💾 تم الحفظ في: {output_path}")
    else:
        cv2.imshow("Result — Press any key to close", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────
#  Argument Parser
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🍪  Face & Eye Detector — Computer Vision Project"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  "-i", help="مسار الصورة (jpg/png)")
    group.add_argument("--camera", "-c", action="store_true",
                       help="استخدام الكاميرا في الوقت الفعلي")

    parser.add_argument("--output", "-o",
                        help="مسار حفظ الصورة الناتجة (اختياري)",
                        default=None)

    args = parser.parse_args()

    if args.camera:
        run_camera()
    else:
        run_image(args.image, args.output)


if __name__ == "__main__":
    main()
