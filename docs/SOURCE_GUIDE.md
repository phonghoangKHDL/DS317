> Cập nhật06/11/2026: nội dung dưới đây mô tả cấu trúc/kiểm tra tại thời điểm bàn giao. Trạng thái thực tế và lệnh chạy hiện tại xem README.md, docs/advisor/STATUS.md và REPOSITORY_REVIEW.md; notebook18/19 đã hoàn tất trên máy.

# Đọc và dùng lại source

## API thường dùng

```python
import baf

frame = baf.load_data(data_path)
metrics = baf.train_classical(
    data_path=data_path,
    output=run_dir,
    methods=baf.BASELINE_METHODS,
    frame=frame,
)

report = baf.audit_results([run_dir])
report.summary
archive = baf.export_proof([run_dir], proof_root)
```

```python
import gru

config = gru.Config(protocol="temporal", mode="pilot")
metrics = gru.train(data_path, run_dir, config=config)

# GRU dùng cùng audit/export với LR và LightGBM.
report = baf.audit_results([run_dir], reference="gru_none")
archive = baf.export_proof([run_dir], proof_root, reference="gru_none")
```

Hai module được chia theo dependency và trách nhiệm. baf.py không import torch, nên03 có thể kiểm tra mọi prediction mà không khởi tạo GPU hoặc cài môi trường huấn luyện GRU. gru.py gộp model và training, gọi lại các hàm dữ liệu/metric/log/proof thay vì có một bộ exporter khác.

Mỗi lần gọi train dùng output mới, không phụ thuộc số notebook hoặc tên file đang mở. methods/seeds/protocol truyền bằng tham số; không có dữ liệu tháng/seed của một lượt được gán cố định trong exporter. Trường expected_rows mặc định1 triệu để tránh dùng nhầm bảng BI; phép ghi đè trong test chỉ dùng dữ liệu tổng hợp có nhãn rõ.

## Trình tự trong baf.py

1. Constants và I/O: load_data, validate_data, temporal_parts, profile.
2. Tính toán dùng chung: clean_features, sample_weights, select_threshold, evaluate.
3. Ghi bằng chứng: experiment, save_manifest, save_predictions, MemoryMonitor.
4. Huấn luyện cổ điển: train_classical và một hàm fit nội bộ.
5. Đọc/so sánh: audit_results, summarize, Comparison.save.
6. Xuất proof chung: export_proof.

## Trình tự trong gru.py

1. Feature list nguồn và Config.
2. TabularGRU: projection/embedding → GRU → dropout → hai logits.
3. prepare: chia dữ liệu, fit preprocessing và tạo tensor một lần cho hai loss.
4. predict: inference theo batch.
5. train: seed/init, trọng số lớp, loop epoch, ghi checkpoints/predictions qua baf.py.

Không có CLI hoặc file vá riêng. Các notebook là nơi cấu hình và thực thi. Copy/paste một hàm ra notebook không cần thiết; gọi API giúp cả bốn notebook dùng cùng logic.

## Điểm cần giữ khi mở rộng

- Thêm loss mới trong cùng vòng training, giữ seed/init/feature nếu muốn đo đóng góp của loss.
- Dùng thư mục output mới và ghi đầy đủ config; đừng tái dùng kết quả cũ với code đã đổi rồi gộp chung.
- export_proof là audit đọc lại, không tự đóng vai trò HPO hoặc kiểm định thống kê.
- Source hợp nhất đã được format88 cột. Thay đổi tên class GRU không làm đổi key state_dict; vẫn đối chiếu toán học bằng checkpoint tác giả được.
