# Rà soát và tinh gọn main —06 /11/2026

Main giữ source/notebook chính, tài liệu phương pháp, bảng kết quả và minh chứng đủ để kiểm chứng. Bản này giảm **1 .107  xuống426 file**, gỡ681 file lặp hoặc phục vụ kiểm tra nội bộ khỏi Git; không viết lại lịch sử và không xóa dữ liệu trên máy.

## Giữ trên GitHub

-20  notebook chính với output; cả4 module `src/`/`supplement/`, dependencies tách profile và các script kiểm chứng. Không thay code huấn luyện, source lock hay môi trường PyTorch CUDA.
- Protocol, phương pháp, nguồn khóa commit/hash, giấy phép/NOTICE, survey và hướng dẫn từng bước05–09 . CSV survey giữ nguyên6.673  byte/CRLF và hash lịch sử.
- Bảng cuối và bảng nghiên cứu gốc;75  selected BAF score/nhãn;30  bộ dự đoán kiểm chứng nhỏ CPU/CUDA, giữ riêng hai môi trường.
- Campaign/suite manifests, FROZEN, validation summary và75 selected COMPLETE records để truy cấu hình/seed/lựa chọn.30  COMPLETE records SmallCNN đi cùng các bộ dự đoán nhỏ.
- Bảng/figure benchmark01–09  và snapshot source lịch sử cần cho 05 /06.

[FILE_INVENTORY.csv](FILE_INVENTORY.csv) hiện chỉ kiểm kê file công khai: path, dung lượng, SHA-256  và lý do giữ. [GITHUB_FILES.txt](docs/repository/GITHUB_FILES.txt) là danh sách chính xác. Inventory tự kiểm dung lượng, để trống hash riêng để tránh tự tham chiếu; mọi file khác bắt buộc có hash. [EVIDENCE_SCOPE.json](docs/repository/EVIDENCE_SCOPE.json) mô tả phạm vi partial, giữ hash của evidence nguồn gốc mà không liệt kê hàng nghìn payload local.

## Gỡ khỏi main, giữ tại máy

- Worker/executed notebook lặp output notebook chính, log từng attempt và log huấn luyện dài: dùng để chẩn đoán/full proof, không cần lặp trên main.
- Record HPO không được chọn, per-epoch/per-weight diagnostics và auxiliary history: giữ local cho phân tích sâu; main giữ validation summary và selected records.
- Source/data snapshots lặp, full-proof hash index và script verify cũ chỉ chạy được khi có payload local: main dùng inventory/verifier hiện tại. Giữ riêng snapshot GRU lịch sử có vai trò thực tế.
- README bàn giao trùng nội dung, synthetic notebook, JSON kiểm tra từng gói, lịch sử cleanup/repair chi tiết và inventory metadata Git: giữ trong hồ sơ local và lịch sử commit. Main giữ kết quả kiểm tra cuối và transition môi trường được manifest tham chiếu.
- Hai bản sao CSV paper table/bootstrap ở advisor_step1 được so byte và xác nhận trùng với bảng canonical trong research evaluation trước khi gỡ.

Trước khi gỡ tracking, đã tạo bản sao có SHA-256  và bảng old path/archive path/reason cho 681  file tại `../BAF_repository_slim_20261106/REMOVED_FROM_MAIN.csv`; bản gốc cũng vẫn ở vị trí cũ và được ignore để không phá đường dẫn resume. Bản inventory đầy đủ5.604  hàng cùng README/review trước sửa nằm trong `../BAF_repository_slim_20261106/before/`. [Inventory đầy đủ ở commit trước](https://github.com/phonghoangKHDL/DS317/blob/e4430e35211e5bd7974b2ffe4c85b2578930287b/FILE_INVENTORY.csv) vẫn truy cập được qua lịch sử Git.

Base.csv, MNIST/CIFAR, model/checkpoint, cache/latent, sampled indices và ZIP proof đầy đủ tiếp tục giữ local. Thư mục `BAF_project_audit_20261031_212825_804674` là hồ sơ dọn cũ, không nhập vào repo.

## Kiểm chứng

Chạy `python -B scripts/verify_repository.py`; thêm `--imports` trong kernel ml-gpu. Verifier giữ nguyên kiểm size/SHA toàn bộ file, notebook/schema,22  fixture nhẹ cho notebook 20 /inventory,75  score BAF,30  bộ dự đoán nhỏ và các bảng mean/SD. `--output` phải nằm ngoài working copy. Kết quả đóng gói ở [VERIFICATION.json](docs/repository/VERIFICATION.json); xác nhận bằng bản xuất mới từ Git index trước commit/push.

Nguồn bốn module, dependencies,20  notebook, mọi score/bảng/manifests/selected records giữ nguyên byte. Không chạy lại HPO, bootstrap hay training. Chỉ stage danh sách chính thức và các entry cần gỡ khỏi index; push bình thường lên origin/main, không force push.

## Giới hạn còn giữ nguyên

-05 /06 cần source snapshot/môi trường cũ và full base run local, không Run All bằng source mở rộng hiện tại.
- Manifest lịch sử chứa đường dẫn tuyệt đối và hash của payload bị lược khỏi GitHub. Đây không phải full proof để resume; train mới cần working copy/campaign mới.
- MWN reference trainer có `cuda(async=True)` không parse trên Python mới; giữ nguyên theo source lock, không import trainer này.
- Tháng7 là development test; image-paper/SmallCNN là adaptation/kiểm chứng nhỏ. LSR budgeted/chung auxiliary seed 42 ; chưa chạy CCRW-E/P. Không gộp CPU/CUDA hoặc coi6 dòng meta tái dùng là6 fit mới.
- Log-loss CPU khi tính lại bằng sklearn mới có sai khác tối đa7,584e-7 ; tolerance 1e-6  đã ghi trong verifier. Accuracy/BAF metrics vẫn kiểm theo quy tắc hiện có.
- Giữ giấy phép/NOTICE theo từng nguồn. Checkout LSR không có LICENSE tìm thấy nên không phân phối lại source snapshot LSR; nhóm chưa gán giấy phép chung cho phần tự viết.
