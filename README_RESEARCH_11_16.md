# BAF: gói khảo sát thực nghiệm weighting 11–16

Giải nén **đè cấu trúc BAF_project hiện tại**. Gói thêm notebook11–16, giữ `src/baf.py` đúng bản09 và mở rộng `src/gru.py`; không chứa notebook01–09 để tránh ghi đè output cũ. Sao lưu hai file src trước khi đè. Module gru vẫn giữ các hàm GRU cũ; không thêm source module thứ ba.

## Cài và mở

1. Đóng kernel cũ. Chạy `install_research.ps1` bằng PowerShell tại project. Script dùng venv `D:\UIT\.venvs\ml-gpu`, cache/temp ở D, giữ bản Torch đang cài. Git for Windows cần có trong PATH. Nếu venv ở vị trí khác, truyền `-PythonExe`.
2. Chọn kernel `Python (UIT ml-gpu)` trong VS Code. Mở notebook11. Sửa `DATA_PATH` tới **Base.csv thật đã dùng notebook09**. Kiểm tra đường dẫn campaign nếu muốn resume.
3. Chạy11 một lần. Lưu lại `CAMPAIGN_READY` được in ra. Mặc định toàn BAF,3seed42/317/2026, cả4suite. Nếu chưa đủ ngân sách LSR, đổi `INCLUDE_LSR=False` **trước khi tạo campaign**. Notebook12–15 tự đọc đúng một campaign; nếu có nhiều campaign phải điền đường dẫn rõ ràng.
4. Run All12 (loss),13 (meta),14 (LSR, nếu chọn),15 (XGB), mỗi notebook một lượt, không chạy đồng thời trên máy12GB RAM. Có thể đổi thứ tự12–15;16 cần tất cả suite đã chọn hoàn tất. Các cell recipe child đang thực thi được lưu trong `attempts/*/training_executed.ipynb`, không yêu cầu Ctrl+S. Nếu lỗi/interrupt, giữ campaign và chạy lại: fit COMPLETE được kiểm hash và reuse, fit dở phải fit lại.
5. Run All16 sau cùng: khóa model/threshold, dự đoán test, xuất CSV, bootstrap, notebook đã execute và proof. Không chạy phương pháp đề xuất ở bước này.

## Kết quả và proof

- `results/research_campaigns/.../suites/*`: mọi recipe/seed, checkpoint, score validation, epoch log, source/data/environment, notebook thực thi thật. Notebook giao diện 12–15 gọi notebook training con; proof chứa notebook con có output trực tiếp của quá trình fit.
- `.../evaluation/research_table.csv`: mean/SD, tách backbone; `test_metrics.csv`: từng seed và fairness; `paired_bootstrap.csv`: ΔAP/CI. Notebook16 cũng xuất `paper_research_table.csv` có tên paper, năm, commit/adaptation và kết quả BAF để điền vào báo cáo. Tất cả số liệu từ các score thực tế, không dùng kết quả synthetic.
- Mỗi suite/stage tạo `*_full_proof.zip` (giữ máy) và `*_review.zip` (gửi để xem nhanh). Review bỏ checkpoint/latent lớn, có danh sách file bị bỏ + hash, giữ selected scores, nhãn, code và notebook output. `verify.ipynb` tự nhận scope partial và chỉ kiểm phần được giữ. Full proof của16 là **proof giai đoạn đánh giá**, không gộp lại model từ mọi suite; để tái kiểm toàn campaign giữ cả full proof12–15 và16.
- Verify là read-only, đọc score/label/hash, không cần CSV gốc hay training lại. Mở thư mục đã giải nén bằng VS Code rồi Run All `verify.ipynb`. Môi trường cần các dependency trong requirements, không cần clone source hay GPU chỉ để verify.
- ZIP gói code không kèm Base.csv, cache, source repo đầy đủ hay model; notebook11 tải nguồn tác giả đúng commit. Không upload dữ liệu/kết quả BAF lên dịch vụ nào tự động.

## Chi phí và phạm vi khoa học

Mặc định20 method/control,3 recipe và3 seed, reuse seed42 đã fit: tổng100 predictor fit cộng LSR auxiliary. Đây có thể là **nhiều giờ hoặc hơn**, đặc biệt LSR; chưa đo runtime trên máy bạn nên không hứa “vài phút”. `mode='pilot'` chỉ kiểm đường chạy trên tập con, không đưa số pilot vào bảng full. Để chuyển pilot→full phải tạo campaign khác, không resume chéo.

Đọc `docs/RESEARCH_METHODS.md` trước khi viết báo cáo. BAF transfers từ image-paper phải ghi rõ adaptation; LSR mặc định giới hạn ngân sách, không gọi tái lập nguyên xi paper/SOTA. LDAM riêng là margin loss; DRW/LDAM-DRW mới chứa weighting. Chưa chạy các temporal paper thiếu source; bảng survey chưa thể gọi bao quát toàn bộ literature.

Gói đã được kiểm bằng dữ liệu **SYNTHETIC** trong `docs/validation/`; đây không phải kết quả BAF. Notebook11–16 phát hành để trống output, chờ chạy trên dữ liệu thật của bạn. Báo cáo kiểm tra kỹ thuật ghi môi trường local; môi trường Windows/GPU của bạn được lưu riêng khi thực thi.
