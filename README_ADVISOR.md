# BAF — bổ sung khảo sát theo góp ý giảng viên (30/10/2026)

Giải nén nội dung ZIP trực tiếp vào `D:\UIT\DS317\BAF_project`.
Gói bổ sung giữ nguyên `src/baf.py`, `src/gru.py`, notebook 01–16 và các campaign cũ.
Hai module tái sử dụng mới nằm trong `supplement/`: `reporting.py`, `training.py`.

## Thứ tự sử dụng

| Notebook | Công việc | Ngân sách |
|---|---|---|
| 17_BAF_advisor_metrics | Audit score và bảng đầy đủ | Không train |
| 18_BAF_CE_batch256 | CE làm đối chứng khớp batch của LRE/MWN | 5 fit MLP trên BAF |
| 19_BAF_sampling_baselines | LightGBM none/balanced/RUS/ROS | 20 fit LightGBM trên BAF |
| 20_Original_dataset_weighting_checks | Công thức/gradient và chạy nhỏ trên MNIST/CIFAR-10 | 6 fit MNIST + 9 fit CIFAR-10; 10 epoch/fit |
| 21_BAF_merge_supplement_tables | Gộp bảng gốc + 18/19 | Không train |

Chọn kernel của `D:\UIT\.venvs\ml-gpu`. Cell đầu in `sys.executable` để đối chiếu.
Nếu thiếu gói nhẹ cho notebook, chạy `scripts/Install-BAFAdvisorDeps.ps1`. Script giữ bản PyTorch đang có.

18 và 19 cần campaign/cache đã tạo ở notebook 11. Nếu chỉ có một campaign, giữ `CAMPAIGN_DIR = None`.
Nếu có nhiều campaign, điền đường dẫn campaign cụ thể vào cell cấu hình. Không chạy lại 11–16.
Cache cũ được kiểm tra hash; các fit mới ghi riêng tại `results/advisor_runs`.

## Bảng đã tổng hợp

`results/advisor_step1/` chứa:

- `test_metrics_verified.csv`: 60 hàng, mỗi hàng một model/seed.
- `all_metrics_mean_sd.csv`: bảng số, 20 phương pháp, trung bình và sample SD.
- `research_table_display.csv`: bảng định dạng để chép sang báo cáo.
- `RESEARCH_TABLE.md`: bảng đọc nhanh theo mô hình nền.
- `group_support.csv`: số mẫu từng nhóm, số negative/positive trong mỗi nhóm.
- `score_audit.csv`, `provenance.json`: nguồn và kết quả đối chiếu 60 file score.

17 đã được chạy với review ZIP gốc. Bản có output nằm ngay trong thư mục notebooks.
Chạy lại 17 với `None` sẽ dùng evaluation trên máy nếu tìm được duy nhất; nếu chưa có, tái tổng hợp CSV đã audit đi kèm.
Có thể đặt `EVALUATION_INPUT` là thư mục `evaluation` hoặc đường dẫn review ZIP.

## Bằng chứng thực thi và chạy tiếp

18–20 thực thi các cell huấn luyện trong tiến trình Python riêng sử dụng đúng kernel hiện tại.
Bản thực thi có output được lưu tại `results/advisor_runs/<run>/attempts/<attempt>/executed.ipynb`.
Output mỗi cell được ghi trước khi parent notebook xuất ZIP; không có bước Ctrl+S rồi chạy lại.

Chạy lại cùng source/config/environment sẽ dùng các fit đã có `COMPLETE.json` và hash hợp lệ.
Fit bị ngắt trước khi hoàn thành được chạy lại; không hứa resume giữa epoch.
Các run thay đổi source/config/environment có thư mục riêng. Không chạy đồng thời hai notebook cùng loại/config.

ZIP `_review.zip` giữ score, bảng, log, notebook và source snapshot; bỏ checkpoint, chỉ số dòng lấy mẫu.
`PROOF_SCOPE.json` liệt kê chính xác phần lược bỏ. Model và chỉ số dòng vẫn nằm ở run local để dùng lại.

## Cách diễn giải kết quả

- 20 là kiểm chứng nhỏ trên dữ liệu của công trình gốc, không phải tái lập đầy đủ điểm số paper.
- 18/19 là các lượt bổ sung chưa có kết quả BAF trong gói này. Chỉ ghi vào báo cáo sau khi chạy xong.
- Resampling là baseline; đóng góp đề xuất của nhóm vẫn thuộc weighting.
- Bảng 21 tách nhóm so sánh theo backbone. LRE/MWN xuất hiện thêm trong nhóm CE batch256 để đối chiếu; không phải hai lượt fit mới.
- Base tháng 7 đã quan sát trong phát triển. Các bảng bổ sung không tạo ra một test độc lập mới.
- SD theo seed không thay CI lấy mẫu test. Gói này không tính lại CI cho các cặp bổ sung.

## Tải dữ liệu của kiểm chứng gốc

MNIST/CIFAR-10 được tải vào `BAF_project/data/external/advisor`, không đóng kèm dataset trong ZIP này.
Code kiểm tra MD5 trước đọc. CIFAR-10 Python archive khoảng 163 MiB; tốc độ tải phụ thuộc mạng.
20 không dùng torchvision hoặc TensorFlow. Có thể đặt `DATASETS = ["mnist"]` hoặc `["cifar10"]` để chạy riêng.
Đó là các cấu hình run khác nhau; chọn cả hai từ đầu nếu muốn một bộ proof chung.
