# Chạy bước 05 — GRU full

Gói cập nhật chỉ thêm notebook 05 và tài liệu của bước này vào `BAF_project`.
Không chép đè notebook 01–04, src, dữ liệu hay các kết quả đang có.
Source vẫn là hai module `baf.py` và `gru.py` của gói BAF_project trước đó.

## Giải nén

ZIP có thư mục gốc `BAF_project`. Chạy PowerShell:

```powershell
Expand-Archive `
  -LiteralPath "$env:USERPROFILE\Downloads\BAF_step05_full_GRU.zip" `
  -DestinationPath "D:\UIT\DS317" -Force
```

Mở `D:\UIT\DS317\BAF_project\notebooks\05_BAF_GRU_full.ipynb`.
Giữ `data/raw/Base.csv` như trước. Nếu giải nén thủ công, gộp thư mục BAF_project
vào project đang có, tránh tạo `BAF_project/BAF_project`.

## Chạy

1. Chọn kernel CPU đã chạy notebook 04. Nếu dùng GPU, chọn kernel venv GPU của bạn.
2. Restart Kernel, kiểm tra DATA_PATH, rồi Run All **một lần**.
3. Cell đầu in `sys.executable`, Torch/CUDA và kiểm tra đúng hai source đã dùng ở pilot.
   Cell trước training kiểm tra đủ một triệu dòng và SHA-256 của Base.csv.
4. Cuối lượt chạy, lưu notebook để giữ output giao diện. Không cần lưu trước cell xuất proof.
5. Gửi notebook có output cùng ZIP mới trong `results/proofs/` để phân tích tiếp.

Notebook không tự tạo venv hay cài lại thư viện. Kernel CPU cũ dùng được.
`device="auto"` chọn CUDA khi kernel đó có CUDA, còn lại dùng CPU.
Đặt `device="cuda"` nếu muốn bắt buộc GPU; không thấy CUDA sẽ báo lỗi.

## Khối lượng và phạm vi

| Mục | Thiết lập |
|---|---|
| Dữ liệu | Base.csv đã dùng ở notebook 01–04, 1.000.000 dòng |
| Protocol | Random stratified 64% train / 16% validation / 20% test |
| Preprocessing | Fit outer train 80% như code tác giả, gồm feature inner validation |
| Model | Numeric projection + categorical embeddings → GRU → 2 logits |
| Kích thước | Embedding 186, hidden 733, dropout 0.3 |
| Optimizer | Adam, lr 0.000448103301026688, batch 256 |
| Lượt chạy | `none` và `balanced`, seed 1234, mỗi loss 8 epoch |
| Checkpoint | Cuối mỗi loss, tổng cộng hai checkpoint |
| Predictions | Validation và random test cho từng loss, tổng cộng bốn file |
| Thời gian CPU | Ngoại suy từ pilot khoảng 17,1 giờ fit cho hai loss; chưa đo full |

Pilot đã chạy trên CPU: 8.192 train, một epoch, 50,052 giây (`none`)
và 48,323 giây (`balanced`), bốn thread. Ngoại suy không tính hết preprocessing,
inference và đóng gói; không áp dụng số này cho GPU hoặc máy khác.
Đây là lượt full dài; giữ máy thức. `EPOCH` chỉ in sau khi xong mỗi epoch,
trên CPU có thể khoảng một giờ mới có dòng tiếp theo.

Code hiện tại chưa có resume giữa epoch. Nếu bị ngắt, log và trạng thái lỗi được giữ;
không coi run đó là hoàn tất. Restart Kernel + Run All tạo thư mục run mới.
Không chạy lại cell training vào RUN_DIR cũ. Có thể chạy lại riêng cell export sau
một run đã hoàn tất nếu thao tác xuất ZIP gặp lỗi, không cần train lại.

## Đọc kết quả và proof

Notebook kiểm tra đủ 640.000/160.000/200.000 dòng, 16 epoch history, bốn predictions,
hai checkpoint và cùng khởi tạo giữa hai loss. Audit tính lại metric từ prediction.
Một seed là bước full đầu tiên, chưa đo độ ổn định giữa seed.

Ngưỡng FPR5% chọn trên validation, sau đó giữ nguyên trên random test.
FPR test thực tế có thể khác 5%. Metric đường ROC trên test chỉ là phân tích,
không dùng để chọn ngưỡng hoặc cấu hình. Bảng 0,5 được hiển thị riêng.
Precision/recall ở 0,5 dùng `score > 0.5` theo evaluator hiện tại.

Random split chứa mọi tháng; không so ngang hàng với temporal LightGBM ở notebook 03.
Đây là chạy lại fixed recipe của paper với paired unweighted ablation được thêm,
không chạy HPO 10 trial hoặc tái tạo toàn bộ lịch sử RNG của tác giả.

Proof xuất một lần chứa log/source/config/preprocessing/split/checkpoint/predictions,
metric, SHA-256 và notebook `verify.ipynb` đã đọc lại kết quả thực tế. Nó không chứa
bản sao giao diện notebook 05 chưa lưu. Sau giải nén proof, mở verify.ipynb và Run All
để audit độc lập; nếu cần, điền PROOF_DIR ở cell đầu của notebook kiểm chứng.

## Nguồn

- Paper: https://www.mdpi.com/2079-3197/13/12/290
- Repo: https://github.com/wn-nwN/fraud-detection-project-Nov2025
- Commit: `6e8252cb9080378905b2aae30033c5946b72335b`
- Bản source gốc và source_manifest đã nằm trong `references/gru_author/` của project.

Xem `docs/STEP05_VALIDATION.json` để biết các kiểm tra gói. Kiểm tra trên dữ liệu tổng hợp
không phải điểm BAF full hoặc số đo GPU của bạn.
