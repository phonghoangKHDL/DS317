> Cập nhật06/11/2026: nội dung dưới đây mô tả cấu trúc/kiểm tra tại thời điểm bàn giao. Trạng thái thực tế và lệnh chạy hiện tại xem README.md, docs/advisor/STATUS.md và REPOSITORY_REVIEW.md; notebook18/19 đã hoàn tất trên máy.

# Thiết lập khảo sát và các đối chứng bổ sung

## Khảo sát đã chạy ở 11–16

- BAF Base: 1.000.000 hồ sơ, 11.029 fraud; 30 thuộc tính hồ sơ, nhãn và month.
- Train gốc tháng 0–5: 794.989 hồ sơ / 8.151 fraud.
- Meta holdout: 256 hồ sơ / 128 fraud, lấy từ tháng 5, seed317, 128 mỗi lớp.
- Core train sau khi loại meta: 794.733 hồ sơ / 8.023 fraud.
- Validation tháng 6: 108.168 hồ sơ / 1.450 fraud.
- Đánh giá phát triển tháng 7: 96.843 hồ sơ / 1.428 fraud; đã quan sát trong các lần phát triển.
- Seed chọn recipe: 42. Recipe được chọn chạy thêm seed317/2026.
- Các tham số điền thiếu, scaler/encoder được fit trên core train. month không nằm trong predictor.

Ngân sách khảo sát gốc là 3 recipe/phương pháp ở seed42, sau đó 2 seed bổ sung cho recipe được chọn.
Tổng 100 predictor fit và 60 selected-model score. LSR dùng chung auxiliary seed42.
Ba recipe là ngân sách giới hạn, không bảo đảm tương đương độ linh hoạt hoặc tối ưu đầy đủ các cơ chế.
Các recipe CB/Focal đổi tham số weighting trong khi CE/LRE/MWN đổi LR. Chênh lệch phải được diễn giải trong phạm vi này.

## CE batch256

MLP hidden (128,64), SGD, momentum0.9, Nesterov, weight decay và epoch/patience kế thừa campaign.
Chỉ đổi batch_size thành256; LR theo ba giá trị của config gốc. Giữ nguyên các dòng core train/meta/validation/test.
Điều này bổ sung đối chứng khớp batch và số cơ hội cập nhật, không thay baseline batch2048 của nhóm loss.

## Sampling LightGBM

Bốn nhóm dùng cùng cache dense, cây tối đa300, learning_rate0.05, min_child_samples100,
L2=1, subsample0.9/subsample_freq1, colsample0.9, deterministic, CPU, số thread từ campaign.
Thử num_leaves15/31/63, AP validation không trọng số và early stopping30 như nhóm LGBM cũ.

- none: toàn bộ core train, không weight.
- balanced: toàn bộ core train, inverse-frequency class weight như đối chứng trước.
- RUS: giữ mọi fraud, lấy ngẫu nhiên một tập con legit để minority/majority=0,1.
- ROS: giữ mọi dòng, lặp thêm fraud để minority/majority≥0,1 (sai khác làm tròn <1 mẫu).

Tỷ lệ0,1 khóa trước chạy, không chọn từ test. RUS/ROS dùng seed tương ứng của fit.
Các nhóm none/balanced cũng xáo thứ tự core train theo seed trong lượt bổ sung; đối chiếu bốn nhóm trong cùng run mới.
Không áp dụng hiệu chỉnh nghịch đảo xác suất, vì mục đích ở đây là baseline resampling thông thường.
Nhóm class-balanced hướng đến mass cân bằng; RUS/ROS có prior khác. Không diễn giải so sánh này là tách riêng cơ chế lấy mẫu ở cùng class mass.

## Độ đo

AP bằng average_precision_score. ROC-AUC, recall benchmark tại FPR≤5% trên ROC test được tính riêng.
Ngưỡng vận hành khóa theo validation FPR≤5%, sau đó áp dụng nguyên giá trị lên test.
FPR/TPR tuổi≥50 và <50, gap older−younger, ratio min/max được báo tại ngưỡng vận hành; giữ số mẫu mỗi nhóm.
SD là sample SD của ba seed, không đo bất định lấy mẫu test.
Không xếp hạng chéo backbone để quy chênh lệch cho weighting.

## Trạng thái bằng chứng

17 đối chiếu trực tiếp 60 score gốc. 18/19 cần thực thi trên cache BAF tại máy nhóm.
20 kiểm chứng nhỏ trên original datasets; giới hạn chi tiết ở IMPLEMENTATION_SCOPE.md.
21 gộp các bảng sau khi 18/19 hoàn tất. Các kết quả mới không dùng để chọn recipe theo test.
