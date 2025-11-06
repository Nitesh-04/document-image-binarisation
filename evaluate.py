import cv2
import numpy as np
import os
import json
import string
import re

# ---------------- Utility Functions ----------------
def oddize(x):
    x = int(round(x))
    if x <= 1:
        return 3
    return x if x % 2 == 1 else x + 1


def compute_band_kernel_sizes(stats, centroids, img_h, num_bands,
                              area_thresh, scale_factor, min_k, max_k, density_shrink_coeff):
    comp_heights, comp_centroids_y = [], []

    for i in range(1, stats.shape[0]):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_thresh:
            continue
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cy = centroids[i][1]
        comp_heights.append(h)
        comp_centroids_y.append(cy)

    if len(comp_heights) == 0:
        return [oddize(min(max(min_k, 5), max_k))] * num_bands

    comp_heights = np.array(comp_heights)
    comp_centroids_y = np.array(comp_centroids_y)
    band_h = img_h / num_bands
    global_med = np.median(comp_heights)
    densities = []
    band_kernel_sizes = []

    for b in range(num_bands):
        y0, y1 = b * band_h, (b + 1) * band_h
        mask = (comp_centroids_y >= y0) & (comp_centroids_y < y1)
        n_in_band = int(mask.sum())

        if n_in_band >= 2:
            med_h = np.median(comp_heights[mask])
            density = n_in_band / band_h
        else:
            med_h = global_med
            density = 0.0

        densities.append(density)
        k = float(med_h) * scale_factor
        shrink_scale = 1.0 / (1.0 + density_shrink_coeff * density * 50.0)
        k *= shrink_scale
        k = max(min_k, min(k, max_k))
        k = oddize(k)
        band_kernel_sizes.append(k)

    smoothed = []
    for i in range(num_bands):
        w_sum, k_sum = 0, 0
        for j in range(max(0, i - 2), min(num_bands, i + 3)):
            w = densities[j] + 0.2
            k_sum += w * band_kernel_sizes[j]
            w_sum += w
        smoothed.append(oddize(k_sum / w_sum))

    return smoothed


def dynamic_closing_by_bands(gray_img, band_kernel_sizes, overlap=0.30):
    h, w = gray_img.shape[:2]
    num_bands = len(band_kernel_sizes)
    band_h = h / num_bands
    out = np.zeros_like(gray_img)

    for b in range(num_bands):
        ksize = int(band_kernel_sizes[b])
        local_overlap = overlap + 0.15 * (ksize / max(band_kernel_sizes))
        start = int(round(max(0, (b - local_overlap) * band_h)))
        end = int(round(min(h, (b + 1 + local_overlap) * band_h)))

        if end <= start:
            continue

        band_slice = gray_img[start:end].copy()
        k_w = oddize(max(3, ksize // 2))
        k_h = oddize(max(3, ksize))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
        closed = cv2.morphologyEx(band_slice, cv2.MORPH_CLOSE, kernel)
        out[start:end] = np.maximum(out[start:end], closed)

    return out

# ---------------- Image Processing ----------------
def process_image_boxes(image_path):
    """Run the pipeline and return final_boxes [(x,y,w,h), ...]"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    num_bands = max(10, min(100, h // 200))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    kernel_init = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bg_init = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel_init)
    diff = cv2.absdiff(gray_eq, bg_init)
    norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, small_k)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, connectivity=8)
    median_area = np.median(stats[1:, cv2.CC_STAT_AREA])
    area_thresh = max(5, median_area * 0.05)
    median_height = np.median(stats[1:, cv2.CC_STAT_HEIGHT])

    scale_factor = min(0.8, max(0.3, median_height / 50.0))
    density_shrink_coeff = 0.6
    min_k = max(3, median_height // 10)
    max_k = max(31, median_height * 3)
    overlap = 0.30
    min_area_keep = max(500, median_area * 0.5)

    band_k = compute_band_kernel_sizes(
        stats, centroids, h, num_bands, area_thresh, scale_factor, min_k, max_k, density_shrink_coeff
    )
    closed_dynamic = dynamic_closing_by_bands(binarized, band_k, overlap)

    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(closed_dynamic, connectivity=8)
    refined = closed_dynamic.copy()
    for i in range(1, num_labels2):
        if stats2[i, cv2.CC_STAT_AREA] < min_area_keep:
            refined[labels2 == i] = 0

    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)

    output = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    boxes = []
    for i in range(1, num_labels2):
        x, y, w, h, area = stats2[i]
        if area >= min_area_keep:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes.append([x, y, w, h])


    output = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    horizontal_gap_threshold = 35
    vertical_overlap_ratio_min = 0.10
    line_grouping_tolerance = 0.5
    

    lines = []
    for box in boxes:
        x, y, w, h = box
        box_center_y = y + h / 2
        placed = False
        
        for line in lines:

            line_y_centers = [b[1] + b[3]/2 for b in line]
            line_heights = [b[3] for b in line]
            line_center_y = sum(line_y_centers) / len(line_y_centers)
            avg_line_height = sum(line_heights) / len(line_heights)
            
            vertical_distance = abs(box_center_y - line_center_y)
            tolerance = avg_line_height * line_grouping_tolerance
            
            if vertical_distance <= tolerance:
                line.append(box)
                placed = True
                break
        
        if not placed:
            lines.append([box])



    final_boxes = []
    for line in lines:
        if len(line) == 0:
            continue
            
        line = sorted(line, key=lambda b: b[0])
        current = line[0]
        
        for next_box in line[1:]:
            x, y, w, h = current
            nx, ny, nw, nh = next_box
            x2, nx2 = x + w, nx + nw

            horizontal_gap = nx - x2
            horizontal_overlap = max(0, x2 - nx)

            vertical_overlap = max(0, min(y + h, ny + nh) - max(y, ny))
            min_h = min(h, nh)
            vertical_overlap_ratio = vertical_overlap / float(min_h) if min_h > 0 else 0

            vertical_ok = vertical_overlap_ratio > vertical_overlap_ratio_min
            horizontal_ok = (horizontal_overlap > 0 or 
                           horizontal_gap <= 0 or 
                           horizontal_gap <= horizontal_gap_threshold)
            
            if vertical_ok and horizontal_ok:
                new_x = min(x, nx)
                new_y = min(y, ny)
                new_x2 = max(x2, nx2)
                new_y2 = max(y + h, ny + nh)
                current = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
            else:
                final_boxes.append(current)
                current = next_box
        final_boxes.append(current)
    
    if len(final_boxes) > 0:
        final_box_areas = [b[2] * b[3] for b in final_boxes]
        avg_area = np.mean(final_box_areas)
        median_area_final = np.median(final_box_areas)
        
        min_area_ratio = 0.1
        min_final_area = avg_area * min_area_ratio
        
        filtered_boxes = [b for b in final_boxes if (b[2] * b[3]) >= min_final_area]
        
        final_boxes = filtered_boxes
    
    final_boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in final_boxes]

    return final_boxes

def numerical_sort(file_list):
    def extract_number(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else -1
    return sorted(file_list, key=extract_number)

def sort_boxes_reading_order(boxes, line_tol=10):
    """
    Sort boxes in top-to-bottom, left-to-right reading order.
    boxes: list of dicts with 'x','y','w','h'
    line_tol: vertical tolerance to consider boxes in the same line
    """
    boxes = sorted(boxes, key=lambda b: b['y'])
    sorted_boxes = []
    while boxes:
        line_y = boxes[0]['y']
        # Collect boxes in the current line
        line = [b for b in boxes if abs(b['y'] - line_y) <= line_tol]
        line = sorted(line, key=lambda b: b['x'])  # sort left-to-right
        sorted_boxes.extend(line)
        boxes = [b for b in boxes if abs(b['y'] - line_y) > line_tol]
    return sorted_boxes


def save_final_boxes_json(image_dir="dataset/Train_Data",
                          output_dir="output_values",
                          num_files=20):
    os.makedirs(output_dir, exist_ok=True)

    # Sort images numerically and pick first num_files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files = numerical_sort(image_files)[:num_files]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing {image_file} ...")
        try:
            # process_image_boxes should return list of [x,y,w,h]
            raw_boxes = process_image_boxes(image_path)
            # Convert to dict format
            boxes_dict = [{'x': int(b[0]), 'y': int(b[1]), 'w': int(b[2]), 'h': int(b[3])} 
                          for b in raw_boxes]
            # Sort in reading order
            boxes_dict = sort_boxes_reading_order(boxes_dict)

            json_file = os.path.splitext(image_file)[0] + ".json"
            json_path = os.path.join(output_dir, json_file)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(boxes_dict, f, indent=2)

        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")


# ---------------- Run ----------------
if __name__ == "__main__":
    save_final_boxes_json()