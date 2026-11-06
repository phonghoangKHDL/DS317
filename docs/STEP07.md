# Bước07: chạy trực tiếp GRU của tác giả

Giải nén vào D:\UIT\DS317, gộp thư mục BAF_project. Gói chỉ thêm notebook07,
các gói nbformat/IPython/tqdm và tài liệu. Không thay src hoặc notebook đã chạy.

## Cài thêm thư viện nhỏ trong venv chung ở D

Chạy PowerShell:

```powershell
Set-Location "D:\UIT\DS317\BAF_project"
New-Item -ItemType Directory -Force -Path "D:\pip-cache", "D:\pip-temp" | Out-Null
$env:PIP_CACHE_DIR = "D:\pip-cache"
$env:TEMP = "D:\pip-temp"
$env:TMP = "D:\pip-temp"
& "D:\UIT\.venvs\ml-gpu\Scripts\python.exe" -m pip install "nbformat>=5.10" "ipython>=8.20" "tqdm>=4.66"
```

Cần Git trên PATH. Mở notebook07, chọn kernel D:\UIT\.venvs\ml-gpu\Scripts\python.exe,
Restart Kernel rồi Run All. Notebook sẽ clone từ GitHub vào external và checkout đúng commit.
Giữ máy thức. Không chạy lại cell training trên cùng run nếu đã hoàn tất.

## Phạm vi và khác biệt được công bố

- Thực thi nguyên cell0,2,3,27,28,29,30,31,33 từ Git blob của commit6e8252cb9080378905b2aae30033c5946b72335b.
- Kiểm tra SHA-256 notebook gốc5391e269bb5388ac5c0dde44662a950de1285e0f5f81c6b02315d466e3422e41.
- Model, Dataset, preprocessing, optimizer, loss, class-weight scan qua shuffled DataLoader giữ nguyên.
- Chỉ đổi đường dẫn CSV tại cell0; thêm observer ghi trial vào cell29, không gọi RNG hay lấy thêm minibatch.
- Tắt thanh tqdm bằng TQDM_DISABLE=1 để notebook output không phình lớn; giữ log Epoch.
- Chạy đủ HPO10 trial69epoch, rồi fixed8epoch:77epoch, seed setup nguyên ở cell29; không reset sau HPO.
- Code testcell33 dùng model fixedcell30. Giữ checkpoint HPOwinner riêng; không tự thay model test bằng nó.
- Không chạy nhánh ML cổ điển, permutation importance hay bootstrap CI. Chỉ là replay pipeline GRU train/tune/test,
  chưa phải toàn bộ thí nghiệm paper. Môi trường thực tế được lưu, repo không có khóa đủ môi trường tác giả.
- Code dùng preprocessing outer80%, có feature inner validation; splitrandom gồm tháng7. Giữ đúng code để đối chiếu.
- Số paper Tables2/3: fraud precision0.05, recall0.78, F10.09; weightedF10.8971, weightedrecall0.83, AUC0.8800.
  Chỉ đối chiếu điểm đã làm tròn, không tự kết luận khớp paper hoặc chỉnh threshold để đuổi số.

Dự trù1–2giờ trên RTX3050 theo khối lượng77epoch và thời gian lượt trước, chưa đo pipeline này.
Dataset gốc, inference và hyperparameter khác có thể làm thời gian khác dự trù. Nếu OOM, giữ log và báo lỗi;
không tự đổi batch/epoch hoặc dùng fallback khác model.

## Bằng chứng

Đây là execution bằng IPython InteractiveShell, không phải một Jupyter kernel thứ hai.
Notebook điều phối07 lưu author_gru_executed.ipynb thật bằng tiến trình Python/IPython riêng cùng sys.executable.
Không cần Ctrl+S trước xuất proof, không cần run2lần. Trạng thái lỗi và notebook partial được lưu nếu bị ngắt.
ZIP proof gồm original source, license/README, git identity/log, cell mapping/diff, executed training notebook,
training log,10HPOtrials,2checkpoint,split IDs/preprocessing,test predictions/metrics,paper comparison vàSHA-256.
verify.ipynb là audit độc lập sau giải nén, không cần rawdata/GPU. Lưu notebook07 khi xong rồi gửi cùng proofZIP.

Kết quả notebook01–06 giữ nguyên. Notebook07 không gọi src/baf.py hoặc src/gru.py.
