# Bảng khảo sát thực nghiệm trên BAF

Các giá trị là trung bình ± sample SD của các seed được ghi ở bảng chi tiết. AP/ROC-AUC ở thang 0–1; recall, precision, FPR, alert rate ở %; gap FPR ở điểm phần trăm; ratio FPR ở thang 0–1; thời gian ở giây.

TPR@FPR≤5% trên ROC test là độ đo benchmark. Recall/FPR frozen dùng ngưỡng đã khóa trên validation; FPR test có thể khác 5%. Chỉ số nhóm tuổi frozen và các ratio/gap được tính riêng từng seed rồi lấy trung bình. SD seed không phải CI lấy mẫu test.

So sánh weighting trong cùng backbone/feature pipeline. CE batch2048 là baseline loss; đối chiếu meta với CE batch256 sau bổ sung. LSR dùng chung auxiliary seed42. Các run này không phải tái lập nguyên thiết lập paper, không xác lập SOTA toàn cục.

## meta_matched_batch256

| method | ap | recall_at_fpr5_curve | fpr_frozen_threshold | older_fpr | younger_fpr | fpr_gap_older_minus_younger |
| --- | --- | --- | --- | --- | --- | --- |
| ce_batch256 | 0.19681 ± 0.00643 | 54.01494 ± 0.87558 | 4.83886 ± 0.41005 | 11.23085 ± 0.60986 | 3.89743 ± 0.41078 | 7.33342 ± 0.48834 |

