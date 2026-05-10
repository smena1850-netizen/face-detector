"""
demo.py — يولّد صورة تجريبية ويشغّل عليها الـ detector
شغّله بـ:  python demo.py
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detect import detect

# ─── نحمّل صورة تجريبية من OpenCV نفسه ───
# (صورة لينا الكلاسيكية اللي بتيجي مع OpenCV)
LENA_PATH = cv2.data.haarcascades.replace("data/", "") + "lena.jpg"

if os.path.exists(LENA_PATH):
    img = cv2.imread(LENA_PATH)
else:
    # لو مش موجودة نعمل صورة اصطناعية بالأبعاد الصح
    print("ℹ️  صورة لينا مش موجودة، هنعمل صورة تجريبية بسيطة...")
    img = np.full((512, 512, 3), 180, dtype=np.uint8)
    # نرسم وجه وهمي بشكل هندسي للتوضيح
    cv2.ellipse(img, (256, 220), (120, 150), 0, 0, 360, (210, 180, 140), -1)  # وجه
    cv2.ellipse(img, (210, 190), (25, 18), 0, 0, 360, (80, 60, 40), -1)       # عين يسار
    cv2.ellipse(img, (302, 190), (25, 18), 0, 0, 360, (80, 60, 40), -1)       # عين يمين
    cv2.ellipse(img, (256, 290), (40, 15), 0, 0, 180, (150, 80, 80), 3)       # فم

result, stats = detect(img)

out_path = os.path.join(os.path.dirname(__file__), "output", "demo_result.jpg")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
cv2.imwrite(out_path, result)

print("─" * 40)
print("  🎉 Demo شغّال بنجاح!")
print(f"  ✅ وجوه    : {stats['faces']}")
print(f"  👁  عيون   : {stats['eyes']}")
print(f"  😊 ابتسامات: {stats['smiles']}")
print(f"  💾 النتيجة : {out_path}")
print("─" * 40)
print("\nلتشغيل على صورتك الخاصة:")
print("  python detect.py --image صورتك.jpg")
print("  python detect.py --image صورتك.jpg --output output/result.jpg")
print("  python detect.py --camera   ← للكاميرا الحية")
