from PIL import Image, ImageDraw, features, __file__ as pil_init
import platform, sys
print("Pillow OK, version:", Image.__version__)
print("PIL.__file__       :", pil_init)
print("Python             :", sys.version)
# print("CPU SIMD           :", features.check_feature('avx2'))