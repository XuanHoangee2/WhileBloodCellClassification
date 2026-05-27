import os
import shutil

folder1 = r"D:\work\WBC_Segmentation\WhileBloodCellClassification\data\SourceData\Dataset"
folder2 = r"D:\work\WBC_Segmentation\WhileBloodCellClassification\data\RawData\Dataset 2"

start_idx = 301

bmp_files = sorted(
    [f for f in os.listdir(folder2) if f.endswith(".bmp")],
    key=lambda x: int(os.path.splitext(x)[0])
)

for i, bmp_file in enumerate(bmp_files):
    old_id = os.path.splitext(bmp_file)[0]   
    new_id = f"{start_idx + i:03d}"          

    bmp_old = os.path.join(folder2, f"{old_id}.bmp")
    png_old = os.path.join(folder2, f"{old_id}.png")

    bmp_new = os.path.join(folder1, f"{new_id}.bmp")
    png_new = os.path.join(folder1, f"{new_id}.png")

    shutil.move(bmp_old, bmp_new)
    shutil.move(png_old, png_new)

print("Done!")