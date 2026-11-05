# Bước 08 — Benchmark Feedzai gốc trên Base

Giải nén ZIP vào `D:\UIT\DS317`. Bên trong ZIP đã có thư mục `BAF_project`.
Gói này thêm notebook 08 và các tài liệu mang tên step08; không thay `src`,
notebook 01–07 hay kết quả đã chạy.

## Chạy

1. Mở `BAF_project/notebooks/08_BAF_Feedzai_original.ipynb` bằng Notebook Editor.
2. Chọn interpreter `D:\UIT\.venvs\ml-gpu\Scripts\python.exe`.
3. Kiểm tra `data/raw/Base.csv` vẫn là bản đã dùng trước đây.
4. Chọn **Restart Kernel → Run All**, để notebook chạy tới cell in đường dẫn ZIP.

Mặc định chạy toàn bộ Base, đủ 100 cấu hình, trên CPU với `n_jobs=10` như tác giả.
Không dùng CUDA. Các cấu hình có thể yêu cầu tới 10.000 cây; không lấy thời gian
chạy trên dữ liệu giả lập để ước lượng thời gian chạy toàn bộ Base.

Notebook tự kiểm tra thư viện. Nếu thiếu các thư viện chính, notebook cài bằng
đúng interpreter đang chọn. Optuna 2.10.0 và phần Group của Aequitas 0.42.0 nằm
riêng trong `external/feedzai08_runtime/packages`, cùng một số phụ thuộc cần
để import chúng. Phần này chỉ dùng sampler và Group, không dùng CLI/dashboard
của hai package. Không cài các phụ thuộc Flask/dashboard cũ của Aequitas vào venv.
Cache và TEMP/TMP của các tiến trình cài đặt nằm dưới project trên ổ D.

Sau khi chạy, giữ ZIP trong `results/proofs` và lưu notebook 08 để tiện đọc.
Không cần lưu notebook thủ công trước khi cell xuất proof chạy. Notebook bằng
chứng được ghi từ output thực sau từng cell trong tiến trình IPython riêng.

## Nguồn và phạm vi

- Repo: https://github.com/feedzai/bank-account-fraud
- Commit: `3377cfd8fc20edcda3dd9dd14c10ed3dd276c6de`.
- Notebook gốc: `notebooks/empirical_results.ipynb`, chạy các cell 0–13.
- Chạy nhánh **Base**; không chạy Type I–V, không tái tạo dữ liệu CTGAN.
- Giữ nguyên YAML, `random_search.py`, 100 seed lấy bằng NumPy seed42,
  và lấy mẫu bằng Optuna **2.10.0** như dependency gốc.
- Không thêm class weighting, early stopping, giảm số cây hoặc đổi không gian HPO.
- Paper và datasheet được lấy trực tiếp từ Git blob của commit và lưu trong proof.

File `feedzai08_parameters.json` là bộ 100 tham số tham chiếu được tạo bằng chính
sampler gốc với Optuna 2.10.0. Lượt chạy thực vẫn thực thi sampler gốc;
file tham chiếu dùng để kiểm chứng từng tham số, không thay sampler.

## Những điểm phải ghi rõ trong báo cáo

**Protocol của tác giả khác notebook 01–03:** train tháng 0–5, test tháng 6–7.
Không có validation riêng trong benchmark này. Toàn bộ feature ngoài nhãn,
bao gồm `month` và `customer_age`, vẫn được đưa vào LightGBM như code gốc.
Năm cột categorical dùng LabelEncoder fit trên train.

Tác giả chọn ROC point có FPR lớn nhất nhưng **nhỏ hơn** 5% trên chính test,
lấy threshold nhỏ nhất và recall lớn nhất tại FPR đó. Sau đó fairness dùng
`score > threshold`, trong khi ROC dùng quy tắc `score >= threshold`.
Hai recall có thể khác khi có điểm bằng threshold; notebook lưu và kiểm tra cả hai.
FPR ratio là `min(FPR Older, FPR Younger) / max(...)`, chia nhóm tuổi ở 50.

Top-five được xếp theo recall trên test, phục vụ mô tả benchmark gốc.
Không dùng top-five này như kết quả chọn hyperparameter độc lập với test.
Tháng 6/7 cũng đã xuất hiện trong các thí nghiệm trước của nhóm, nên không gọi
đây là bộ test hoàn toàn chưa từng được xem.

### Các thay đổi tương thích được công khai

1. Chỉ đọc Base từ CSV thay cho sáu file parquet.
2. Code gọi Aequitas bằng `score_col`/`label_col`, nhưng Group 0.42.0 không nhận
   hai keyword đó. Lớp tương thích đổi tên cột về `score`/`label_value` và ép cột
   nhóm sang object; công thức và implementation Group vẫn là Aequitas 0.42.0.
3. `Series.iteritems()` đổi thành `items()` để dùng pandas hiện đại.
4. Ghim NumPy seed42 trước shuffle train/test. Notebook gốc không ghim RNG cho
   hai lần shuffle này; seed42 trong cell HPO chỉ xuất hiện sau bước shuffle.
5. Dùng LightGBM hiện tại trong venv, được ghi lại trong `runtime_versions.json`.
   Máy đã dùng LightGBM 4.7.0, khác bản 3.1.1 của tác giả. Không tuyên bố tái tạo
   nguyên môi trường Python3.7 hay bảo đảm khớp số trong paper.
6. Thêm timer, lưu evidence sau từng trial và giảm log lặp, không sửa tham số model.

Source gốc, hash từng cell, diff và tất cả version thực tế nằm trong proof.
Đây là **chạy lại nhánh Base của code tác giả với các sửa đổi tương thích**.

## Proof

Mỗi trial có model LightGBM dạng text gzip, dự đoán float64 dạng NPZ và JSON
tham số/metric. Các trial dùng chung file thứ tự split có row index, nhãn, tháng
và tuổi. Không đóng gói raw CSV vào ZIP.

Ngoài ra có notebook thực thi các cell tác giả, training log, bảng tổng hợp,
top-five mô tả trên test, biểu đồ Base, giấy phép, paper/datasheet, source diff,
`verify.ipynb`, trạng thái export và SHA-256 cho mọi file trong gói proof.

Để kiểm lại sau khi giải nén, mở `verify.ipynb` trong thư mục proof rồi Run All.
Không cần GPU, raw CSV hoặc train lại. Audit tính lại metric và fairness từ
dự đoán, kiểm tra tham số/seed, thứ tự chia dữ liệu, số cây và checksum.
Audit này không thay thế việc chạy inference từ checkpoint trên raw CSV.

Khi bị ngắt, các trial đã xong vẫn nằm trong thư mục run. Phiên bản này không
gộp tự động các lượt bị ngắt: Restart Kernel + Run All tạo run mới.

## Tiếp nối cho phần cải thiện

Giữ lượt tái lập này làm đối chiếu. Thí nghiệm class weighting tiếp theo phải
chọn trọng số/ngưỡng trên validation, giữ cùng ngân sách HPO giữa các phương pháp,
và báo rõ lịch sử đã xem test. Không so trực tiếp hai metric có split hoặc
quy tắc threshold khác nhau như một kết quả cải thiện.

Kết quả kiểm thử của người tạo gói nằm trong `STEP08_VALIDATION.json`:
dữ liệu giả lập, không phải kết quả chạy toàn bộ Base trên máy bạn.
