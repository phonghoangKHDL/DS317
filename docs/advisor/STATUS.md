# Trạng thái bàn giao — 30/10/2026

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| 1. Bảng độ đo BAF | Đã chạy và đối chiếu 60 score / 20 phương pháp | Notebook17, results/advisor_step1 |
| 2. CE batch256 trên BAF | Đã viết, kiểm tra pipeline; chưa chạy full BAF | Notebook18; cần cache gốc tại máy nhóm |
| 3. Sampling baselines BAF | Đã viết, kiểm tra pipeline; chưa chạy full BAF | Notebook19; cần cache gốc tại máy nhóm |
| 4a. Công thức/source | Đã chạy 8 kiểm tra số học, xác minh source hash | checks/numerical_checks.csv và source_hash_checks.csv |
| 4b. Dữ liệu gốc quy mô nhỏ | Đã chạy 15 fit, 3 seed, MNIST và CIFAR-10 | Notebook20 có output, original_dataset_metrics.csv |

## Kết quả kiểm chứng nhỏ

- cifar10 / cb_ce: accuracy 0.32427 ± 0.01077.
- cifar10 / ce: accuracy 0.32313 ± 0.00457.
- cifar10 / mwn: accuracy 0.32587 ± 0.00692.
- mnist / ce: accuracy 0.49322 ± 0.00000.
- mnist / lre: accuracy 0.78018 ± 0.14683.

Các số liệu này thuộc kiểm chứng SmallCNN với ngân sách 10 epoch, không phải điểm số paper gốc.
MNIST CE dự đoán thiên về lớp lớn; LRE có SD lớn. CIFAR có độ chính xác thấp ở ngân sách nhỏ.
Đó là giới hạn quan sát được của kiểm chứng này, không phải bằng chứng đã tái lập đầy đủ paper.
Không suy ra tính đúng của toàn bộ chương trình chỉ từ các số accuracy này.
Các cột fit_seconds trong kiểm chứng nhỏ đo thời gian vòng fit/checkpoint và dự đoán test của lượt đó.

## Trình tự trên máy nhóm

1. Giải nén vào BAF_project.
2. Chạy18 (5 fit), chạy19 (20 fit), dùng đúng campaign của11–16.
3. Chạy21 để gộp bảng. Có thể chạy lại17 để tự audit score gốc tại máy.
4.20 đã có proof thực thi; chạy lại nếu muốn kiểm tra trong môi trường của nhóm.

Chưa chạy CCRW-E hoặc thí nghiệm cải tiến ở giai đoạn này.
