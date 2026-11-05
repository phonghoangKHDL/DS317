# Bước 09 — Chọn class weighting bằng validation

## Cài vào project đang dùng

Giải nén ZIP vào **`D:\UIT\DS317`**. ZIP đã chứa thư mục `BAF_project`.
Gói thêm notebook09, tài liệu/requirements riêng, và cập nhật `src/baf.py`.
Không thay `src/gru.py`, data, notebook01–08 hoặc các run đã lưu.
Project vẫn dùng hai module chính `baf.py` và `gru.py`.

Mở `notebooks/09_BAF_class_weight_selection.ipynb`, chọn
`D:\UIT\.venvs\ml-gpu\Scripts\python.exe`, rồi **Restart Kernel → Run All**.
Mặc định chạy CPU với 4 thread. GPU không tham gia lượt này.
Kernel đang giữ source cũ sẽ được reload, và notebook kiểm tra hash source trước khi chạy.

Các thư viện chính đã dùng ở bước trước. Nếu thiếu, notebook cài bằng đúng
interpreter đang chọn, với cache/TEMP/TMP dưới project trên ổ D.
Notebook09 không dùng runtime Optuna/Aequitas của bước08 và không thay PyTorch.

## Thí nghiệm

| Thành phần | Thiết lập |
| --- | --- |
| Train | Tháng 0–5 |
| Validation | Tháng 6 |
| Test | Tháng 7 |
| Phương pháp | Không trọng số, balanced, trọng số được chọn bằng validation |
| HPO | 24 candidate mỗi phương pháp, seed42 |
| Ngân sách | 72 fit HPO; cùng candidate model và cùng số cây yêu cầu |
| Cây | Mỗi candidate yêu cầu 150, 300, 600 hoặc 1.000 cây |
| Chọn cấu hình | AP validation cao nhất; hòa thì candidate index nhỏ nhất |
| Evaluation seeds | 42, 317, 2026; dùng lại fit seed42 đã chọn |
| Fit bổ sung | Tối đa 6 model cho seed317/2026 |
| Ngưỡng | Chọn riêng trên validation của mỗi model, FPR ≤5% |
| Test | Chỉ tính metric sau khi cố định toàn bộ 9 model/ngưỡng |

Kế hoạch mặc định yêu cầu **14.800 cây cho mỗi phương pháp**, tổng 44.400 cây
HPO. Model có thể dừng tăng cây khi không còn split hợp lệ theo LightGBM;
notebook không thêm early stopping và không thay số cây theo test.
Thời gian chạy còn phụ thuộc cấu hình và máy; không dùng thời gian kiểm thử
trên dữ liệu giả lập để dự báo thời gian chạy toàn bộ Base.

Các candidate được lập trước bằng NumPy seed9026. Cả ba phương pháp dùng cùng
24 cấu hình model. Candidate đầu là recipe LightGBM đã dùng trong notebook01–03;
các candidate còn lại tìm learning rate, độ sâu, số lá, số cây, số mẫu/lá,
subsample, colsample và L2 trong phạm vi cố định.
Không dùng thứ hạng test hoặc tham số trial84 từ notebook08.

Phương pháp `tuned` ghép mỗi candidate với **một** power trong kế hoạch đã chốt.
Đây là 24 lần thử kết hợp model/power, không phải 24 × 7 lần thử.
`none` và `balanced` cũng có đúng 24 lần thử. Số seed thay đổi ngẫu nhiên trong
model; cách chia theo tháng và các hàng dữ liệu không thay đổi giữa seed.

## Trọng số

Từ nhãn train, đặt `r = n_negative / n_positive` và
`w_positive / w_negative = r ** power`. Sau đó chuẩn hóa để trung bình trọng số
trên train bằng 1.

- Power0 cho trọng số `[1, 1]`, khớp `none`.
- Power1 cho `n / (2*n_class)`, khớp `balanced` của scikit-learn.
- Kế hoạch dùng các power: 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5.

Trọng số được truyền qua `sample_weight`. Không cộng thêm `class_weight`,
`is_unbalance` hay `scale_pos_weight`. Chuẩn hóa giúp tránh việc thay tổng
trọng số toàn cục gây thêm khác biệt về regularization ngoài tỷ lệ hai lớp.
Đây là tuning trọng số cho thí nghiệm môn học, không tự nhận là thuật toán mới.

## Preprocessing và khác biệt với bước08

Giữ pipeline của notebook01–03: loại `month` và nhãn khỏi feature; xử lý sentinel
`-1` ở năm cột đã ghi trong source, intended balance âm là missing, numeric
float32; năm cột categorical dùng vocabulary chỉ lấy từ train.
Validation/test có giá trị categorical chưa gặp được ánh xạ thành missing.

CSV được đọc để tạo các partition. Quá trình HPO không nhận feature/nhãn test
để fit, chọn candidate hay threshold. Test không được truyền làm `eval_set`.

Bước08 dùng train0–5/test6–7, giữ feature và quy tắc threshold của Feedzai.
Bước09 dùng train0–5/validation6/test7 và pipeline nhóm đã dùng từ bước01.
Vì vậy không so trực tiếp score ở08 với09 để tuyên bố cải thiện.

**Lịch sử holdout:** tháng7 đã xuất hiện ở các notebook trước, đặc biệt07/08.
Kết quả09 là kiểm tra tiếp trên test đã từng được xem, không phải một holdout
hoàn toàn mới. Việc cố định quy trình bây giờ không xóa lịch sử đó.

## Theo dõi tiến độ và resume

Output hiển thị nhóm phương pháp, candidate, seed, số cây và thời gian đã chạy.
Heartbeat mỗi 15 giây cũng hoạt động trong giai đoạn tạo histogram trước cây đầu.
`DONE` nghĩa là model, dự đoán validation và JSON record đã hoàn tất.
`CACHE` nghĩa là record đã được kiểm tra và dùng lại.

Mặc định mỗi Run All tạo thư mục `weighting_study_<time>_<uuid>` mới.
Nếu bị ngắt:

1. Giữ nguyên thư mục run đã in ở cell cấu hình.
2. Điền đường dẫn đó vào `RESUME_RUN`.
3. Giữ nguyên CONFIG, data, source và phiên bản thư viện.
4. Restart Kernel → Run All.

Run đang dở chỉ dùng lại các fit có record hoàn chỉnh và artifact/hash hợp lệ.
Fit chưa hoàn tất sẽ chạy lại. Thay dữ liệu, candidate plan, thread, version,
source hoặc tham số sẽ bị từ chối thay vì trộn hai thí nghiệm.
Run đã hoàn tất được giữ lại và có thể kiểm lại/xuất lại proof, không train thêm.

## Cố định lựa chọn trước test

`selection.json` ghi ba cấu hình thắng theo AP validation ở seed42.
Giữ ba fit đó, fit thêm hai seed cho từng cấu hình đã chọn, rồi lấy threshold
riêng trên validation của từng model. Không chọn lại hyperparameter theo seed.

`frozen_plan.json` ghi đủ chín cặp method/seed, threshold, checkpoint,
validation score, hash split và preprocessing. `freeze_seal.json` khóa hash
file này. Source kiểm tra các ràng buộc đó trước khi tính test.
`events.jsonl` ghi thứ tự selection → freeze toàn bộ threshold → test.

Test dùng `score >= threshold`. Không tìm threshold mới bằng nhãn test.
FPR test có thể vượt 5% do thay đổi phân phối. Báo **recall tại ngưỡng cố định
từ validation** cùng FPR thực tế; không gọi đó là recall tại đúng FPR5% trên test.

## Đọc kết quả

- `selections.csv`: candidate/power được chọn bằng validation cho mỗi phương pháp.
- `metrics.csv`: AP, AUC, precision, recall, FPR, alert rate và confusion counts
  của validation/test cho chín model.
- `summary.csv`: mean và sample std giữa các seed, tách validation/test.
- `deltas.csv`: chênh lệch từng seed so với `none` trên cùng test.
- `test_comparison.png`: AP, recall và FPR trên test, error bar là sample std.

AP là chỉ số chính đã chốt. Recall/FPR phải đọc cùng nhau. Sample std giữa ba
seed không phải khoảng tin cậy và không đo toàn bộ độ bất định do lấy mẫu data.
Seed42 cũng là seed dùng chọn HPO; ba seed không phải ba lượt HPO độc lập.

Nếu `tuned` chọn power0, ghi nhận validation đã chọn không tái trọng số ở
candidate đó. Nếu AP test không tăng, giữ kết luận không cải thiện; không chọn
lại power, candidate hay seed sau khi xem test.

## Proof và mã nguồn

Cell export chỉ audit và đóng gói, không train lại, không cần Ctrl+S giữa chừng.
Proof nằm trong `results/proofs/<run_name>_proof.zip`, bao gồm:

- source snapshot, protocol, candidate plan và split/preprocessing;
- model text gzip và dự đoán float64 NPZ cho tất cả fit;
- lựa chọn validation, freeze/seal, dự đoán test và bảng metric;
- notebook thực thi thật và log của mọi lần chạy trong `evidence/executions`;
- bản lần chạy gần nhất `study_executed.ipynb`;
- biểu đồ, `verify.ipynb`, trạng thái export và SHA-256.

Giải nén proof, mở `verify.ipynb` từ thư mục proof và Run All để tính lại metric.
Không cần raw CSV, GPU hay train. Checkpoint được kiểm tra hash và số cây;
đọc lại proof không thay thế việc chạy inference từ checkpoint trên raw CSV.

Mã mới được thêm vào cuối `src/baf.py`; phần huấn luyện/audit cũ được giữ nguyên.
Hash toàn bộ file đổi vì có API mới. Các notebook cũ ghim toàn bộ source hash
cần source snapshot tương ứng để chạy lại đúng version; không thay hash guard
chỉ để ép chạy. Các run/proof cũ đã lưu vẫn giữ nguyên.

`STEP09_VALIDATION.json` ghi kiểm thử bằng dữ liệu giả lập của người tạo gói,
không phải kết quả full Base trên máy bạn.

## Script giải nén bằng PowerShell

```powershell
$downloadFolder = Join-Path $env:USERPROFILE "Downloads"
$zipFile = Get-ChildItem -LiteralPath $downloadFolder `
    -Filter "BAF_step09_class_weighting*.zip" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $zipFile) {
    throw "Chưa thấy ZIP bước09 trong Downloads."
}

Expand-Archive -LiteralPath $zipFile.FullName `
    -DestinationPath "D:\UIT\DS317" -Force
```
