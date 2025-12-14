# import torch
# import os
# import json
# import logging
# import paddle
# from paddleocr import PaddleOCR
# import cv2
# import numpy as np
# import Config

# # ==========================================
# # 1. CẤU HÌNH ĐƯỜNG DẪN
# # ==========================================

# Task_2_config = Config.returnTestTask2_Config()

# INPUT_IMG_DIR = Task_2_config["input_images"] 
# INPUT_JSON_DIR = Task_2_config["input_json"] 
# OUTPUT_DIR = Task_2_config["output"] 

# # Các đuôi ảnh hỗ trợ
# VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# # Tắt log rác của Paddle
# logging.getLogger("ppocr").setLevel(logging.WARNING)

# # ==========================================
# # 2. CÁC HÀM TIỆN ÍCH (CẮT & XOAY ẢNH)
# # ==========================================

# def parse_polygon_from_dict(poly_dict):
#     """Chuyển đổi dict {x0, y0...} sang numpy array"""
#     return np.array([
#         [poly_dict["x0"], poly_dict["y0"]],
#         [poly_dict["x1"], poly_dict["y1"]],
#         [poly_dict["x2"], poly_dict["y2"]],
#         [poly_dict["x3"], poly_dict["y3"]]
#     ], dtype=np.float32)

# def sorted_boxes(dt_boxes):
#     """
#     [FIX] Hàm này bị thiếu trong code của bạn.
#     Sắp xếp lại thứ tự 4 điểm: TL, TR, BR, BL
#     """
#     num_points = dt_boxes.shape[0]
#     sorted_points = sorted(dt_boxes, key=lambda x: x[0])
#     left_points = sorted_points[:2]
#     right_points = sorted_points[2:]

#     if left_points[0][1] < left_points[1][1]:
#         tl = left_points[0]
#         bl = left_points[1]
#     else:
#         tl = left_points[1]
#         bl = left_points[0]

#     if right_points[0][1] < right_points[1][1]:
#         tr = right_points[0]
#         br = right_points[1]
#     else:
#         tr = right_points[1]
#         br = right_points[0]
    
#     return np.array([tl, tr, br, bl], dtype=np.float32)

# def get_rotate_crop_image(img, points):
#     """Cắt và xoay ảnh theo 4 điểm (fix nghiêng)"""
#     # Tính chiều rộng và cao của box mới
#     width_top = np.linalg.norm(points[0] - points[1])
#     width_bottom = np.linalg.norm(points[2] - points[3])
#     max_width = int(max(width_top, width_bottom))

#     height_left = np.linalg.norm(points[0] - points[3])
#     height_right = np.linalg.norm(points[1] - points[2])
#     max_height = int(max(height_left, height_right))

#     # Điểm đích
#     dst_pts = np.array([
#         [0, 0],
#         [max_width - 1, 0],
#         [max_width - 1, max_height - 1],
#         [0, max_height - 1]
#     ], dtype=np.float32)

#     # Biến đổi và cắt
#     M = cv2.getPerspectiveTransform(points, dst_pts)
#     dst_img = cv2.warpPerspective(img, M, (max_width, max_height))
#     return dst_img

# def read_image_windows(path):
#     """Đọc ảnh hỗ trợ đường dẫn tiếng Việt/Unicode"""
#     if not os.path.exists(path):
#         print(f"  [ERR] File không tồn tại: {path}")
#         return None
#     try:
#         stream = np.fromfile(path, dtype=np.uint8)
#         img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)
#         if img_bgr is None: return None
#         return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     except Exception as e:
#         print(f"  [ERR] Lỗi đọc ảnh: {e}")
#         return None

# def expand_polygon(points, img_height, img_width, scale_ratio=1.1):
#     """Nới rộng polygon từ tâm ra các phía."""
#     center = np.mean(points, axis=0)
#     vectors = points - center
#     expanded_points = center + vectors * scale_ratio
    
#     # Clip tọa độ
#     expanded_points[:, 0] = np.clip(expanded_points[:, 0], 0, img_width - 1)
#     expanded_points[:, 1] = np.clip(expanded_points[:, 1], 0, img_height - 1)
    
#     return expanded_points.astype(np.float32)

# # ==========================================
# # 3. KHỞI TẠO MODEL
# # ==========================================
# def init_model():
#     print("--- Đang khởi tạo Model PaddleOCR ---")
#     try:
#         if paddle.is_compiled_with_cuda():
#             paddle.device.set_device("gpu")
#             print(" -> [OK] Đã kích hoạt chế độ GPU.")
#         else:
#             paddle.device.set_device("cpu")
#             print(" -> [WARN] Chạy trên CPU.")
#     except Exception:
#         pass

#     model = PaddleOCR(
#         lang="en",
#         use_textline_orientation=True,
#         use_doc_orientation_classify=False,
#         use_doc_unwarping=False,
#     )
#     return model

# # ==========================================
# # 4. XỬ LÝ CHÍNH
# # ==========================================
# def process_single_image_yolo(ocr_model, img_path, json_path):
#     img = read_image_windows(img_path)
#     if img is None: return [], None

#     if not os.path.exists(json_path):
#         return [], img
    
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception:
#         return [], img

#     if "task2" not in data or "output" not in data["task2"]:
#         return [], img
    
#     input_blocks = data["task2"]["output"]["text_blocks"]
#     final_blocks = []

#     img_h, img_w = img.shape[:2]

#     # Tạo folder debug để kiểm tra xem model nhìn thấy gì (Quan trọng)
#     debug_dir = "debug_crops"
#     if not os.path.exists(debug_dir): os.makedirs(debug_dir)

#     for idx, block in enumerate(input_blocks):
#         try:
#             poly_dict = block["polygon"]
#             poly_points = parse_polygon_from_dict(poly_dict)

#             # ---------------------------------------------------------
#             # CHECK 1: Kiểm tra xem tọa độ có bị chuẩn hóa (0.0 - 1.0) không?
#             # Nếu tọa độ toàn < 1.0 nghĩa là đang sai hệ quy chiếu
#             if np.max(poly_points) <= 1.5: 
#                 print(f"[WARN] Tọa độ có vẻ là Normalized. Đang scale lại theo ảnh {img_w}x{img_h}")
#                 poly_points[:, 0] *= img_w
#                 poly_points[:, 1] *= img_h
#             # ---------------------------------------------------------

#             # 1. Nới rộng box (Box Dilation) - Giữ mức 1.1 là an toàn
#             poly_points = expand_polygon(poly_points, img_h, img_w, scale_ratio=1.1)
            
#             # 2. Sắp xếp điểm
#             poly_points = sorted_boxes(poly_points)
            
#             # 3. Cắt ảnh
#             crop_img = get_rotate_crop_image(img, poly_points)

#             if crop_img is None or crop_img.shape[0] < 4 or crop_img.shape[1] < 4:
#                 final_blocks.append(block)
#                 continue

#             # C. Lưu ảnh ra để debug (Xóa dòng này khi đã chạy ngon)
#             cv2.imwrite(f"{debug_dir}/crop_{idx}.jpg", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

#             # 4. Gọi OCR
#             crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
#             rec_results = ocr_model.ocr(crop_bgr)

#             rec_text = ""
#             rec_score = 0.0

#             if rec_results and isinstance(rec_results, list) and len(rec_results) > 0:
#                 # Lấy khối kết quả đầu tiên (là một list chứa các kết quả từng dòng)
#                 first_result = rec_results[0]
                
#                 # PaddleOCR trả về một list chứa các cặp [tọa độ, (text, score)]
#                 if isinstance(first_result, list):
                    
#                     all_text = []
#                     max_score = 0.0
                    
#                     for item in first_result:
#                         # item là một list, ví dụ: [ [1, 1], [1, 8], ('9', 0.4278) ]
#                         # Lấy phần tử cuối cùng, là tuple ('Text', Score)
#                         if isinstance(item, list) and len(item) > 1 and isinstance(item[-1], tuple):
#                             text, score = item[-1]
                            
#                             # Gom text lại
#                             all_text.append(text)
                            
#                             # Cập nhật score cao nhất
#                             score = float(score)
#                             if score > max_score:
#                                 max_score = score
                                
#                     # Gán kết quả cuối cùng
#                     rec_text = " ".join(all_text)
#                     rec_score = max_score

#             # print(rec_results)

#             # In ra màn hình để xem nó có đọc được không
#             # print(f" - Block {idx}: '{rec_text}' ({rec_score:.2f})")

#             new_block = block.copy()
#             new_block["text"] = rec_text
#             new_block["score"] = round(float(rec_score), 4)
#             final_blocks.append(new_block)

#         except Exception as e:
#             print(f"Lỗi block {idx}: {e}")
#             final_blocks.append(block)

#     return final_blocks, img

# # ==========================================
# # 5. VISUALIZATION & SAVE
# # ==========================================

# def visualize_and_save(img_rgb, blocks, filename, output_dir):
#     if img_rgb is None or not blocks: return
#     vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
#     for b in blocks:
#         poly = b["polygon"]
#         pts = np.array([
#             [poly["x0"], poly["y0"]], [poly["x1"], poly["y1"]],
#             [poly["x2"], poly["y2"]], [poly["x3"], poly["y3"]]
#         ], np.int32).reshape((-1, 1, 2))
        
#         cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
#         if b.get("text"):
#             # Lưu ý: cv2.putText không hỗ trợ tiếng Việt có dấu
#             cv2.putText(vis_img, b["text"], (pts[0][0][0], pts[0][0][1] - 5), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

#     save_path = os.path.join(output_dir, filename)
#     cv2.imwrite(save_path, vis_img)

# def save_json(data, output_path):
#     final_output = {
#         "task2": {
#             "input": {"task1_output": {"chart_type": "vertical bar"}},
#             "name": "Text Detection and Recognition",
#             "output": {"text_blocks": data},
#         }
#     }
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, ensure_ascii=False, indent=4)

# # ==========================================
# # 6. MAIN
# # ==========================================
# def main():
#     if not os.path.exists(INPUT_IMG_DIR):
#         print(f"[LỖI] Thư mục ảnh không tồn tại: {INPUT_IMG_DIR}")
#         return
#     if not os.path.exists(INPUT_JSON_DIR):
#         print(f"[LỖI] Thư mục JSON YOLO không tồn tại: {INPUT_JSON_DIR}")
#         return

#     vis_output_dir = os.path.join(OUTPUT_DIR, "visualize_images")
#     if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
#     if not os.path.exists(vis_output_dir): os.makedirs(vis_output_dir)

#     ocr = init_model()

#     files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
#     total = len(files)

#     print(f"\nTìm thấy {total} ảnh.")
#     print("-" * 50)

#     for idx, filename in enumerate(files):
#         img_path = os.path.join(INPUT_IMG_DIR, filename)
        
#         json_name = os.path.splitext(filename)[0] + ".json"
#         json_input_path = os.path.join(INPUT_JSON_DIR, json_name)
#         json_output_path = os.path.join(OUTPUT_DIR, json_name)

#         print(f"[{idx + 1}/{total}] Đang xử lý: {filename}")

#         blocks, img_rgb = process_single_image_yolo(ocr, img_path, json_input_path)
        
#         if blocks:
#             save_json(blocks, json_output_path)

#         if img_rgb is not None and len(blocks) > 0:
#             visualize_and_save(img_rgb, blocks, filename, vis_output_dir)

#     print("-" * 50)
#     print("Hoàn tất.")
#     print(f"File JSON kết quả tại: {OUTPUT_DIR}")
#     print(f"Ảnh Visualize tại: {vis_output_dir}")

# -*- coding: utf-8 -*-
"""
Task2 (YOLO -> PaddleOCR 3.x Recognition-only)

- Input: Ảnh + JSON từ YOLO (task2/output/text_blocks với polygon x0..y3)
- Output: JSON bổ sung text + score cho từng block + ảnh visualize + debug crops

Lưu ý quan trọng (PaddleOCR 3.x):
- PaddleOCR.ocr() không còn tham số det/rec. Nếu bạn muốn recognition-only sau YOLO,
  hãy dùng module TextRecognition (và TextLineOrientationClassification nếu cần).
"""

import os
import json
import logging

import cv2
import numpy as np
import paddle

import Config

# PaddleOCR 3.x modules
from paddleocr import TextRecognition, TextLineOrientationClassification


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

Task_2_config = Config.returnTestTask2_Config()

INPUT_IMG_DIR = Task_2_config["input_images"]
INPUT_JSON_DIR = Task_2_config["input_json"]
OUTPUT_DIR = Task_2_config["output"]

# Các đuôi ảnh hỗ trợ
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# Tắt log rác của PaddleOCR
logging.getLogger("ppocr").setLevel(logging.WARNING)


# ==========================================
# 2. THAM SỐ TIỀN XỬ LÝ CROP (khuyến nghị bật)
# ==========================================

POLY_DILATE_RATIO = 1.10   # nới rộng polygon (box dilation)
UPSCALE_MIN_H = 32         # nếu crop thấp hơn -> upscale để OCR dễ đọc hơn
UPSCALE_MAX_SCALE = 4.0
ENABLE_UPSCALE = True

PAD_PX = 10                # padding chống sát viền (thường cải thiện rec)
ENABLE_PADDING = True

# Lưu crop debug (để kiểm tra YOLO polygon / crop)
SAVE_DEBUG_CROPS = True


# ==========================================
# 3. CÁC HÀM TIỆN ÍCH (CẮT & XOAY ẢNH)
# ==========================================

def parse_polygon_from_dict(poly_dict):
    """Chuyển dict {x0,y0..x3,y3} -> np.array shape (4,2), float32"""
    return np.array([
        [poly_dict["x0"], poly_dict["y0"]],
        [poly_dict["x1"], poly_dict["y1"]],
        [poly_dict["x2"], poly_dict["y2"]],
        [poly_dict["x3"], poly_dict["y3"]],
    ], dtype=np.float32)


def sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
    """
    Sắp xếp lại 4 điểm theo thứ tự: TL, TR, BR, BL.
    dt_boxes: (4,2)
    """
    pts = dt_boxes.tolist()
    pts_sorted_x = sorted(pts, key=lambda p: p[0])
    left = pts_sorted_x[:2]
    right = pts_sorted_x[2:]

    left = sorted(left, key=lambda p: p[1])   # top -> bottom
    right = sorted(right, key=lambda p: p[1]) # top -> bottom

    tl, bl = left[0], left[1]
    tr, br = right[0], right[1]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_rotate_crop_image(img_rgb: np.ndarray, points_4x2: np.ndarray) -> np.ndarray:
    """Cắt và warp theo 4 điểm (fix nghiêng)."""
    # width
    width_top = np.linalg.norm(points_4x2[0] - points_4x2[1])
    width_bottom = np.linalg.norm(points_4x2[2] - points_4x2[3])
    max_width = int(max(width_top, width_bottom))

    # height
    height_left = np.linalg.norm(points_4x2[0] - points_4x2[3])
    height_right = np.linalg.norm(points_4x2[1] - points_4x2[2])
    max_height = int(max(height_left, height_right))

    if max_width < 2 or max_height < 2:
        return None

    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(points_4x2, dst_pts)
    return cv2.warpPerspective(img_rgb, M, (max_width, max_height))


def read_image_windows(path: str):
    """Đọc ảnh hỗ trợ đường dẫn tiếng Việt/Unicode (Windows). Trả về RGB."""
    if not os.path.exists(path):
        print(f"  [ERR] File không tồn tại: {path}")
        return None
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  [ERR] Lỗi đọc ảnh: {e}")
        return None


def expand_polygon(points: np.ndarray, img_height: int, img_width: int, scale_ratio: float = 1.1):
    """Nới rộng polygon từ tâm ra các phía."""
    center = np.mean(points, axis=0)
    vectors = points - center
    expanded = center + vectors * scale_ratio

    expanded[:, 0] = np.clip(expanded[:, 0], 0, img_width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, img_height - 1)
    return expanded.astype(np.float32)


def preprocess_crop_for_rec(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý crop trước recognition:
    - Upscale nếu quá nhỏ
    - Padding chống sát viền
    Return: crop_rgb đã xử lý
    """
    out = crop_rgb

    if ENABLE_UPSCALE:
        h, w = out.shape[:2]
        if h > 0 and h < UPSCALE_MIN_H:
            scale = UPSCALE_MIN_H / float(h)
            if scale > UPSCALE_MAX_SCALE:
                scale = UPSCALE_MAX_SCALE
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if ENABLE_PADDING:
        out = cv2.copyMakeBorder(
            out,
            top=PAD_PX, bottom=PAD_PX, left=PAD_PX, right=PAD_PX,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
    return out


# ==========================================
# 4. WRAPPER: YOLO -> (ORI optional) -> REC
# ==========================================

class YoloPaddleRecognizer:
    def __init__(
        self,
        rec_model_name: str = "PP-OCRv5_server_rec",
        use_textline_orientation: bool = True,
        ori_model_name: str = "PP-LCNet_x0_25_textline_ori",
    ):
        # Device
        if paddle.is_compiled_with_cuda():
            self.device = "gpu:0"
            try:
                paddle.device.set_device(self.device)
            except Exception:
                pass
            print(" -> [OK] PaddlePaddle GPU enabled.")
        else:
            self.device = "cpu"
            try:
                paddle.device.set_device(self.device)
            except Exception:
                pass
            print(" -> [WARN] PaddlePaddle running on CPU.")

        # Recognition model
        self.rec = TextRecognition(model_name=rec_model_name, device=self.device)

        # Optional textline orientation classifier
        self.use_ori = use_textline_orientation
        self.ori = None
        if self.use_ori:
            try:
                self.ori = TextLineOrientationClassification(model_name=ori_model_name, device=self.device)
            except Exception as e:
                print(f" -> [WARN] Không khởi tạo được TextLineOrientationClassification: {e}")
                self.ori = None
                self.use_ori = False

    def recognize(self, crop_rgb: np.ndarray):
        """
        Input: crop RGB (numpy)
        Output: (text, score, extra_info)
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return "", 0.0, {"reason": "empty_crop"}

        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        # 1) Orientation (0/180) nếu bật
        ori_label = None
        ori_score = None
        if self.use_ori and self.ori is not None:
            try:
                ori_out = self.ori.predict(input=crop_bgr, batch_size=1)
                if ori_out:
                    j = ori_out[0].json
                    label_names = j.get("res", {}).get("label_names", None)
                    scores = j.get("res", {}).get("scores", None)
                    if isinstance(label_names, list) and len(label_names) > 0:
                        ori_label = label_names[0]
                    if scores is not None and len(scores) > 0:
                        ori_score = float(scores[0])
                    if ori_label == "180_degree":
                        crop_bgr = cv2.rotate(crop_bgr, cv2.ROTATE_180)
            except Exception:
                ori_label = None
                ori_score = None

        # 2) Recognition
        try:
            rec_out = self.rec.predict(input=crop_bgr, batch_size=1)
            if not rec_out:
                return "", 0.0, {"ori_label": ori_label, "ori_score": ori_score, "reason": "empty_rec_out"}

            j = rec_out[0].json
            rec_text = j.get("res", {}).get("rec_text", "") or ""
            rec_score = float(j.get("res", {}).get("rec_score", 0.0) or 0.0)

            return rec_text, rec_score, {"ori_label": ori_label, "ori_score": ori_score}
        except Exception as e:
            return "", 0.0, {"ori_label": ori_label, "ori_score": ori_score, "error": str(e)}


# ==========================================
# 5. KHỞI TẠO MODEL
# ==========================================

def init_model():
    print("--- Khởi tạo PaddleOCR 3.x (Recognition-only) ---")
    # Nếu bạn không truy cập được HuggingFace để tải model:
    # os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    return YoloPaddleRecognizer(
        rec_model_name="PP-OCRv5_server_rec",
        use_textline_orientation=True,
        ori_model_name="PP-LCNet_x0_25_textline_ori",
    )


# ==========================================
# 6. XỬ LÝ CHÍNH (YOLO JSON -> REC)
# ==========================================

def process_single_image_yolo(ocr_model: YoloPaddleRecognizer, img_path: str, json_path: str):
    img = read_image_windows(img_path)
    if img is None:
        return [], None

    if not os.path.exists(json_path):
        return [], img

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], img

    if "task2" not in data or "output" not in data["task2"]:
        return [], img

    input_blocks = data["task2"]["output"].get("text_blocks", [])
    final_blocks = []

    img_h, img_w = img.shape[:2]

    # Debug dir theo từng ảnh để tránh overwrite
    filename_stem = os.path.splitext(os.path.basename(img_path))[0]
    debug_dir = os.path.join(OUTPUT_DIR, "debug_crops", filename_stem)
    if SAVE_DEBUG_CROPS:
        os.makedirs(debug_dir, exist_ok=True)

    for idx, block in enumerate(input_blocks):
        try:
            poly_dict = block["polygon"]
            poly_points = parse_polygon_from_dict(poly_dict)

            # CHECK: Normalized coordinates?
            if np.max(poly_points) <= 1.5:
                print(f"[WARN] Polygon có vẻ normalized. Scale theo ảnh {img_w}x{img_h}")
                poly_points[:, 0] *= img_w
                poly_points[:, 1] *= img_h

            # 1) Dilation
            poly_points = expand_polygon(poly_points, img_h, img_w, scale_ratio=POLY_DILATE_RATIO)

            # 2) Sort 4 points
            poly_points = sorted_boxes(poly_points)

            # 3) Crop (RGB)
            crop_img = get_rotate_crop_image(img, poly_points)

            if crop_img is None or crop_img.shape[0] < 4 or crop_img.shape[1] < 4:
                final_blocks.append(block)
                continue

            # 4) Preprocess crop (upscale/pad)
            crop_img = preprocess_crop_for_rec(crop_img)

            # 5) Save debug crop
            if SAVE_DEBUG_CROPS:
                crop_bgr_dbg = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(debug_dir, f"crop_{idx:04d}.jpg"), crop_bgr_dbg)

            # 6) Recognition-only
            rec_text, rec_score, _extra = ocr_model.recognize(crop_img)

            print(f" - Block {idx}: '{rec_text}' ({rec_score:.2f})")

            new_block = block.copy()
            new_block["text"] = rec_text
            new_block["score"] = round(float(rec_score), 4)
            final_blocks.append(new_block)

        except Exception as e:
            print(f"[ERR] Lỗi block {idx}: {e}")
            final_blocks.append(block)

    return final_blocks, img


# ==========================================
# 7. VISUALIZATION & SAVE
# ==========================================

def visualize_and_save(img_rgb, blocks, filename, output_dir):
    if img_rgb is None or not blocks:
        return

    vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    for b in blocks:
        poly = b["polygon"]
        pts = np.array([
            [poly["x0"], poly["y0"]],
            [poly["x1"], poly["y1"]],
            [poly["x2"], poly["y2"]],
            [poly["x3"], poly["y3"]],
        ], np.int32).reshape((-1, 1, 2))

        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        if b.get("text"):
            cv2.putText(
                vis_img,
                str(b["text"]),
                (int(pts[0][0][0]), int(pts[0][0][1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, vis_img)


def save_json(data_blocks, output_path):
    final_output = {
        "task2": {
            "input": {"task1_output": {"chart_type": "vertical bar"}},
            "name": "Text Detection and Recognition",
            "output": {"text_blocks": data_blocks},
        }
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)


# ==========================================
# 8. MAIN
# ==========================================

def main():
    if not os.path.exists(INPUT_IMG_DIR):
        print(f"[LỖI] Thư mục ảnh không tồn tại: {INPUT_IMG_DIR}")
        return
    if not os.path.exists(INPUT_JSON_DIR):
        print(f"[LỖI] Thư mục JSON YOLO không tồn tại: {INPUT_JSON_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vis_output_dir = os.path.join(OUTPUT_DIR, "visualize_images")
    os.makedirs(vis_output_dir, exist_ok=True)

    ocr = init_model()

    files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    total = len(files)

    print(f"\nTìm thấy {total} ảnh.")
    print("-" * 50)

    for idx, filename in enumerate(files):
        img_path = os.path.join(INPUT_IMG_DIR, filename)

        json_name = os.path.splitext(filename)[0] + ".json"
        json_input_path = os.path.join(INPUT_JSON_DIR, json_name)
        json_output_path = os.path.join(OUTPUT_DIR, json_name)

        print(f"[{idx + 1}/{total}] Đang xử lý: {filename}")

        blocks, img_rgb = process_single_image_yolo(ocr, img_path, json_input_path)

        if blocks:
            save_json(blocks, json_output_path)

        if img_rgb is not None and len(blocks) > 0:
            visualize_and_save(img_rgb, blocks, filename, vis_output_dir)

    print("-" * 50)
    print("Hoàn tất.")
    print(f"File JSON kết quả tại: {OUTPUT_DIR}")
    print(f"Ảnh Visualize tại: {vis_output_dir}")


if __name__ == "__main__":
    main()
