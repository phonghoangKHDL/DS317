# Bổ sung theo góp ý giảng viên — đối chiếu 06/11/2026

Hướng dẫn hiện tại: [README](README.md). Không cần giải nén gói bổ sung vào repo đã clone.

| Notebook | Trạng thái thực tế | Bằng chứng |
|---|---|---|
| 17 | Đã audit60 score/20 phương pháp | `results/advisor_step1/` |
| 18 | Hoàn tất5 fit,3 selected scores | `results/advisor_runs/ce_0c139aa5b451/` |
| 19 | Hoàn tất20 fit,12 selected scores | `results/advisor_runs/sampling_1827c85af012/` |
| 20 | Hoàn tất15 fit CUDA sau attempt lỗi CSV;15 fit CPU bàn giao giữ riêng | `original_checks_582e4a8ada1a/` và `original_checks_3848ef4489af/` dưới `results/advisor_runs/` |
| 21 | Đã ghép81 dòng/27 nhóm so sánh | `results/advisor_combined/` |

81 dòng gồm75 score duy nhất và6 dòng LRE/MWN tái dùng, không phải6 fit mới. Không trộn số liệu CPU/CUDA của20. Notebook20 hiện tại có patch retry CSV; không có notebook 22 cần giữ. `src/` và `supplement/` giữ nguyên hash khớp run.

Nguồn CB/LRE/MWN, LICENSE/NOTICE và hash vẫn ở `docs/advisor/upstream/` và `source_lock.json`. Kiểm chứng SmallCNN10 epoch không phải tái lập đầy đủ paper. GitHub giữ score/nhãn/bảng/config/fit records/notebook-log. Checkpoint/cache/sampling indices/ZIP proof giữ local theo `docs/repository/EVIDENCE_SCOPE.json`.

Tháng7 đã xem khi phát triển; chưa chạy CCRW-E/P. CI gốc không tự bao phủ18/19.

Notebook20 đã bỏ run mặc định cố định trong log. `RESUME_RUN = None` chọn theo identity hiện tại; explicit resume chỉ nhận manifest khớp hoàn toàn tại đường dẫn canonical. Luồng chọn được kiểm tra nhẹ; manifest, output và evidence run cũ không bị sửa.
