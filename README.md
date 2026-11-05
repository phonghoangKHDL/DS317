# BAF project — bản source gọn và dùng lại được

Project mới cho notebook01–04. Source chỉ có hai module theo chức năng, không có các file patch/export riêng cho từng notebook. Mỗi câu lệnh được viết riêng và format theo Black, line length88.

| File | Trách nhiệm |
|---|---|
| src/baf.py | Đọc/kiểm tra dữ liệu, tiền xử lý, LR/LightGBM, metric, audit/so sánh và xuất proof dùng chung |
| src/gru.py | Cấu hình, model GRU, chuẩn bị tensor và huấn luyện hai loss; dùng lại metric/log/proof từ baf.py |
| notebooks/01_BAF_class_weighting.ipynb | 8 lượt LR/LightGBM không trọng số/balanced |
| notebooks/02_BAF_cost_weighting.ipynb | 9 lượt LightGBM theo3 giá trị lambda |
| notebooks/03_BAF_comparison.ipynb | Đọc và kiểm tra kết quả01/02, ghép seed, phân tích fraud đổi quyết định |
| notebooks/04_BAF_GRU_reproduction.ipynb | GRU fixed recipe tác giả / temporal adaptation; mặc định pilot |

## Giải nén và giữ công việc đang có

ZIP chứa thư mục **BAF_project_clean**, nên giải nén bên cạnh BAF_project cũ. Đừng đổi tên hoặc chép đè notebook đã chạy. Project mới đọc lại các thư mục kết quả cũ được, không cần huấn luyện lại chỉ vì refactor.

```powershell
Expand-Archive `
  -LiteralPath "$env:USERPROFILE\Downloads\BAF_project_clean.zip" `
  -DestinationPath "D:\UIT\DS317" -Force
```

Mở D:\UIT\DS317\BAF_project_clean bằng VS Code.

## Môi trường

Từ project root, tạo venv nếu chưa có môi trường phù hợp:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chọn Select Kernel → Python Environments → .venv. Các thư viện tối thiểu được ghi trong requirements; phiên bản thực tế được lưu trong từng manifest.

Notebook04 cần thêm **torch**. Dùng https://pytorch.org/get-started/locally/ chọn Windows / Pip / Python / CUDA phù hợp driver; chạy lệnh của selector bằng `.\.venv\Scripts\python.exe -m pip` thay pip3. Chỉ cần torch. Cài torch trước khi bắt đầu các lượt so sánh nếu muốn giữ cùng môi trường; hoặc dùng môi trường GRU riêng và ghi rõ khác biệt.

## Dữ liệu và kết quả cũ

- Đặt Base.csv đầy đủ trong data/raw, hoặc sửa DATA_PATH trong notebook.
- Nếu hai project cùng nằm trong D:\UIT\DS317 và Base.csv còn ở BAF_project/data/raw, notebook tự dùng lại đường dẫn đó; không sao chép dữ liệu.
- Notebook03 ưu tiên results/runs của project mới. Nếu chưa có baseline mới, nó tìm BAF_project/results/runs bên cạnh. Nếu có nhiều lượt, điền BASELINE_DIR và COST_DIR để chọn; không tự lấy lượt mới nhất.
- Có thể mở notebook03 ngay để đọc17 lượt cũ đã chạy. Không cần chạy01/02 lại.
- Kết quả01/02 cũ phải được so với nhau; kết quả mới phải cùng phiên bản source. Audit từ chối trộn source/data/protocol khác nhau.
- GRU cũ được audit riêng bằng `baf.audit_results([duong_dan_gru])`; không gộp random-split GRU vào bảng temporal01/02.

## Cách chạy và proof

Sau khi điền cấu hình, mỗi notebook chạy từ trên xuống **một lượt**. Cell cuối gọi cùng hàm:

```python
archive = baf.export_proof(
    run_paths=[RUN_DIR],
    output_root=ROOT / "results/proofs",
)
```

Notebook03 đưa hai run_paths và reference="lgbm_balanced". Notebook04 dùng reference="gru_none". Không cần Ctrl+S trước cell xuất proof, không cần gọi lại cell để đợi autosave.

Proof chứa một notebook verify.ipynb được thực thi thật từ predictions/log đã lưu, snapshot source, config, metric, checkpoint nếu có và SHA-256. Đây là notebook **kiểm chứng đọc lại**, không phải bản sao nguyên giao diện chưa lưu. Log huấn luyện gốc vẫn được giữ. Xuất proof không huấn luyện thêm.

Sau khi giải nén proof, mở verify.ipynb trong thư mục đó và Run All để kiểm tra độc lập. Nếu mở từ working directory khác, điền PROOF_DIR ở cell đầu. Không cần torch để kiểm tra predictions GRU. Lưu notebook làm việc khi kết thúc để giữ output giao diện và các chỉnh sửa cấu hình.

## Phạm vi thực nghiệm

- Giữ cấu hình LR/LightGBM và quy tắc missing của bản cũ; balanced vẫn N/(2*n_c), cost vẫn w0=lambda,w1=1, không tự chuẩn hóa lại.
- Train0–5/validation6 cho01/02. Không dùng tháng7 trong các lượt đó.
- GRU model/state_dict/forward được giữ về tính toán, gộp model/trainer vào gru.py và đổi tên class thành TabularGRU. Kiến trúc dùng các cột của từng hồ sơ làm chuỗi, không phải chuỗi giao dịch theo thời gian.
- author_fixed giữ recipe random64/16/20 và preprocessing outer80% như code nguồn, có feature của internal validation; gồm mọi tháng. Không chạy lại HPO10 trial hoặc toàn lịch sử RNG tác giả.
- temporal GRU là adaptation train0–5/validation6, train-only preprocessing; không đánh giá7.
- Pilot GRU chỉ train mẫu nhỏ một epoch để đo tài nguyên. Không dùng điểm pilot làm benchmark, không gọi các cấu hình đơn giản là các thuật toán SOTA độc lập.
- Std giữa seed chưa phải khoảng tin cậy; không có bootstrap trong bản này.
- Month7 đã được xem trong notebook nhóm trước đây; đánh giá tương lai vẫn cần công bố lịch sử đó.

## Kiểm tra của bản refactor

Xem results/checks/refactor_verification.json. Phân biệt ba loại bằng chứng:

1. Audit lại17 file prediction BAF thật đã được cung cấp; không huấn luyện lại BAF.
2. Kiểm tra LR/LightGBM mới và bản gốc cho cùng predictions trên dữ liệu tổng hợp, cùng môi trường/cấu hình.
3. Kiểm tra model GRU mới so với model tác giả cho cùng logits/gradients, và chạy notebook từ đầu trên dữ liệu tổng hợp.

Các kiểm tra tổng hợp không phải điểm BAF hoặc bằng chứng VRAM6GB. Chưa chạy benchmark BAF mới/CUDA trong môi trường tạo gói.

references/survey_20261026 giữ nguyên bảng khảo sát cũ để truy nguồn; các lệnh trong báo cáo lịch sử đó thuộc cấu trúc cũ. Dùng README hiện tại cho cách chạy. references/gru_author chứa source snapshot/commit để đối chiếu kiến trúc.
