# Kiểm tra bản refactor

Kết quả máy đọc: results/checks/refactor_verification.json.

Đã thực hiện12 kiểm tra. Đây là kiểm tra phần mềm và đối chiếu predictions đã có, không phải lượt benchmark BAF mới.

| Nhóm | Bằng chứng |
|---|---|
| Cấu trúc source | Chỉ baf.py và gru.py; parse/compile được; không có câu lệnh ghép bằng dấu chấm phẩy |
| Kết quả BAF cũ | Đọc17 prediction files đã cung cấp, tính lại metric; mean/std khớp bảng cũ trong tolerance1e-12 |
| LR/LightGBM | Cùng dữ liệu tổng hợp, cùng tham số/môi trường: predictions của nhánh balanced khớp chính xác code gốc |
| Kiến trúc GRU | Embedding186/hidden733; nạp cùng state_dict: logits và gradients khớp chính xác model tác giả |
| Notebook01–04 | Mỗi notebook chạy mọi code cell đúng một lượt, tiến trình Python/IPython mới, dữ liệu tổng hợp gắn nhãn rõ, tham số nhỏ để kiểm tra luồng |
| Proof độc lập | Giải nén sang thư mục khác rồi chạy lại verify.ipynb từ tiến trình sạch |
| GRU temporal | Hai loss chạy/ghi epoch/checkpoint, chỉ validation6; export cùng hàm dùng cho các backbone khác |
| Tính toàn vẹn | Thay nhãn trong prediction làm audit thất bại |
| Dùng lại exporter | Gọi nhiều lần tạo các gói mới, không ghi đè dữ liệu cũ |

Notebook được kiểm tra bằng cách thực thi từng cell qua IPython trong một tiến trình mới, không phải thao tác trực tiếp trên giao diện VS Code/Windows. Môi trường kiểm tra không hỗ trợ socket để khởi chạy Jupyter kernel ngoài tiến trình; không dùng điều đó làm lý do bỏ kiểm tra thực thi code.

Các trường hợp dữ liệu nhỏ dùng đầy đủ schema BAF nhưng không phải BAF thật, được ghi dataset_tag=SYNTHETIC_VERIFICATION_ONLY. Cấu hình nhỏ chỉ áp dụng trong bản notebook dùng kiểm tra; notebook giao cho người dùng vẫn có mặc định1 triệu dòng và cấu hình baseline/GRU đã công bố.

Chưa huấn luyện BAF mới hoặc kiểm tra CUDA/VRAM6GB. Phần đo RSS có thể không khả dụng trong môi trường hạn chế tiến trình; khi đó lưu null và memory_warning, không điền số0 hoặc làm dừng huấn luyện. Trên máy người dùng, số đo thực tế vẫn được lưu nếu psutil đọc được.

## Tự kiểm tra các kết quả đang có

Trong notebook, chọn đúng thư mục run rồi gọi:

```python
report = baf.audit_results([BASELINE_DIR, COST_DIR], reference="lgbm_balanced")
display(report.summary)
display(report.prediction_audit)
```

Với GRU dùng một RUN_DIR và reference="gru_none". Hàm kiểm tra source hash, số lượt, row IDs/nhãn, metric, ngưỡng, lịch sử epoch/checkpoint GRU. Không import hoặc chạy source huấn luyện cũ khi audit.

Để kiểm chứng độc lập khi trao đổi kết quả, xuất proof và dùng verify.ipynb trong ZIP. Bộ code kiểm chứng và dữ liệu dự đoán cần thiết đều nằm trong gói đó.
