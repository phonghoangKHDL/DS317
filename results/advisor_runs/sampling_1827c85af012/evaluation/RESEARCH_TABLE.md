# Bảng khảo sát thực nghiệm trên BAF

Các giá trị là trung bình ± sample SD của các seed được ghi ở bảng chi tiết. AP/ROC-AUC ở thang 0–1; recall, precision, FPR, alert rate ở %; gap FPR ở điểm phần trăm; ratio FPR ở thang 0–1; thời gian ở giây.

TPR@FPR≤5% trên ROC test là độ đo benchmark. Recall/FPR frozen dùng ngưỡng đã khóa trên validation; FPR test có thể khác 5%. Chỉ số nhóm tuổi frozen và các ratio/gap được tính riêng từng seed rồi lấy trung bình. SD seed không phải CI lấy mẫu test.

So sánh weighting trong cùng backbone/feature pipeline. CE batch2048 là baseline loss; đối chiếu meta với CE batch256 sau bổ sung. LSR dùng chung auxiliary seed42. Các run này không phải tái lập nguyên thiết lập paper, không xác lập SOTA toàn cục.

## sampling_matched_lightgbm

| method | ap | recall_at_fpr5_curve | fpr_frozen_threshold | older_fpr | younger_fpr | fpr_gap_older_minus_younger |
| --- | --- | --- | --- | --- | --- | --- |
| lgbm_balanced | 0.21622 ± 0.00250 | 57.39963 ± 0.22511 | 4.13003 ± 0.30174 | 10.01442 ± 0.75534 | 3.26335 ± 0.23539 | 6.75107 ± 0.52203 |
| lgbm_none | 0.21327 ± 0.00176 | 57.32960 ± 0.45563 | 4.34139 ± 0.07296 | 10.75462 ± 0.04921 | 3.39682 ± 0.08661 | 7.35780 ± 0.11676 |
| lgbm_ros | 0.21470 ± 0.00102 | 57.39963 ± 0.35247 | 4.09684 ± 0.03644 | 10.08246 ± 0.22551 | 3.21526 ± 0.03341 | 6.86720 ± 0.23488 |
| lgbm_rus | 0.21459 ± 0.00111 | 56.93277 ± 0.28011 | 4.56183 ± 0.11950 | 10.76006 ± 0.30688 | 3.64893 ± 0.12501 | 7.11114 ± 0.31960 |

