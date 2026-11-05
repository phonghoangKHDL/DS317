# Bước 06 — hoàn thiện GRU 3 seed

Giải nén BAF_step06_three_seeds.zip vào D:\UIT\DS317, gộp BAF_project với project hiện có.
Gói cập nhật hai module src/baf.py và src/gru.py, thêm notebook06. Giữ notebook01–05 và kết quả cũ.

Mở notebooks/06_BAF_GRU_three_seeds.ipynb, chọn kernel D:\UIT\.venvs\ml-gpu\Scripts\python.exe,
Restart Kernel, rồi Run All một lượt. Không cài thêm hoặc nâng cấp thư viện.
BASE_RUN_DIR mặc định đã trỏ đúng thư mục full seed1234 vừa chạy. Nếu chỉ còn proof,
giải nén proof cũ rồi sửa BASE_RUN_DIR trỏ tới inputs/run_01. Không cần train lại1234.

Seed mới317/2026, mỗi seed2loss ×8epoch:32epoch mới, dự trù45phút fit theo lượt đo vừa rồi.
Tổng hợp cả1234 thành3seed,48epoch,6checkpoint,12prediction. Mean/std nằm trong operating_points.csv
của proof. Std là sample standard deviation, không phải khoảng tin cậy. Seed thay đổi khởi tạo và
minibatch, split vẫn cố định42. Không chọn cấu hình/seed theo random test.

Source GRU chỉ sửa to_numpy(copy=True) ở nhãn. Audit cho ghép các seed rời nhau khi dữ liệu,
toàn bộ row IDs/split/preprocessing, packages/Python/GPU và mọi config khác seed đều giống nhau.
Source baf trước audit_results phải trùng nguyên văn; source gru chỉ được khác copy=True.
Mỗi run vẫn giữ source snapshot và hash gốc, không sửa manifest/hash của seed1234.
Mặc định audit/export cũ vẫn giữ kiểm tra strict; ghép seed dùng allow_seed_extension=True.

Notebook05 đã ghim hash source cũ; giữ nó làm bằng chứng đã chạy. Dùng notebook06 với source mới.
Không dùng notebook05 cũ để chạy lại sau khi thay source. Notebook01–04 vẫn gọi API mặc định.
Proof verify.ipynb được thực thi từ kết quả lưu trên đĩa, không phải bản sao giao diện chưa lưu.
Không cần Ctrl+S trước export. Sau khi xong, lưu notebook06 và gửi notebook cùng ZIP proof mới.

Không ghép random-split GRU với temporal LightGBM. Cấu hình tác giả giữ preprocessing outer80%,
có feature inner validation; random split gồm tháng7. Không chạy lại HPO10 trial.
