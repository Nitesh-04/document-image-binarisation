import cv2
import numpy as np

def oddize(x):
    x = int(round(x))
    if x <= 1:
        return 3
    return x if x % 2 == 1 else x + 1

def compute_band_kernel_sizes(stats, centroids, img_h,
                              num_bands,
                              area_thresh,
                              scale_factor,
                              min_k, max_k,
                              density_shrink_coeff):
    comp_heights = []
    comp_centroids_y = []
    n = stats.shape[0]
    for i in range(1, n):
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

    band_kernel_sizes = []
    band_h = img_h / num_bands
    global_med = np.median(comp_heights)
    densities = []

    for b in range(num_bands):
        y0 = b * band_h
        y1 = (b + 1) * band_h
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
        k = k * shrink_scale
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


def main():
    image_path = "dataset/Train_Data/6306.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Failed to read '{image_path}' — check file path.")

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

    band_k = compute_band_kernel_sizes(stats, centroids, binarized.shape[0],
                                      num_bands=num_bands,
                                      area_thresh=area_thresh,
                                      scale_factor=scale_factor,
                                      min_k=min_k, max_k=max_k,
                                      density_shrink_coeff=density_shrink_coeff)

    print("All band kernel sizes:", band_k)

    closed_dynamic = dynamic_closing_by_bands(binarized, band_k, overlap=overlap)

    num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(closed_dynamic, connectivity=8)
    refined = closed_dynamic.copy()
    for i in range(1, num_labels2):
        if stats2[i, cv2.CC_STAT_AREA] < min_area_keep:
            refined[labels2 == i] = 0

    output = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    boxes = []
    for i in range(1, num_labels2):
        x, y, w, h, area = stats2[i]
        if area >= min_area_keep:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes.append([x, y, w, h])

    print("Initial box count before merging:", len(boxes))

    output = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    horizontal_gap_threshold = 20
    vertical_overlap_ratio_min = 0
    line_grouping_tolerance = 1.0
    

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
    
    print(f"\nGrouped into {len(lines)} lines:")


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

    print("Final merged box count:", len(final_boxes))

    if len(final_boxes) > 0:
        final_box_areas = [b[2] * b[3] for b in final_boxes]
        avg_area = np.mean(final_box_areas)
        median_area_final = np.median(final_box_areas)
        
        min_area_ratio = 0.1
        min_final_area = avg_area * min_area_ratio
        
        filtered_boxes = [b for b in final_boxes if (b[2] * b[3]) >= min_final_area]
        
        removed_count = len(final_boxes) - len(filtered_boxes)
        if removed_count > 0:
            print(f"\nFiltered out {removed_count} very small boxes:")
            print(f"  Average final box area: {avg_area:.0f}px²")
            print(f"  Median final box area: {median_area_final:.0f}px²")
            print(f"  Min area threshold (5% of avg): {min_final_area:.0f}px²")
        
        final_boxes = filtered_boxes
        print(f"Final box count after small box filtering: {len(final_boxes)}")

    for (x, y, w, h) in final_boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2)

    finalimg = img.copy()
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(finalimg, (x, y), (x + w, y + h), (0, 0, 0), 5)
    cv2.imshow("Final Detected Text Lines", finalimg)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# Contrast normalization (CLAHE).
# Estimate background via large closing, get foreground via image difference.
# Smooth & Otsu threshold → binary image.
# Small opening to remove specks.
# Connected components to compute statistics (size/centroid).
# Compute per-band kernel sizes based on component heights and density. (core adaptive logic)
# Apply closing per horizontal band using per-band kernels (with overlap) and blend results.
# Post-process: connected components + filter by area → draw boxes.