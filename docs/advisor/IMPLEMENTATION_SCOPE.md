# Phạm vi kiểm chứng

## Nguồn gốc

Các file tác giả đi kèm giữ nguyên byte từ proof cũ; hash đối chiếu `source_lock.json`.
Giấy phép/NOTICE của từng repository được giữ bên cạnh source.

| Cơ chế | Nguồn | Cách sử dụng trong kiểm chứng |
|---|---|---|
| Class-Balanced | richardaecn/class-balanced-loss, src/cifar_main.py | Kiểm tra effective-number, loss và gradient; gọi port hiện có trên CIFAR-10 |
| LRE | uber-research/learning-to-reweight-examples, mnist/reweight.py | Kiểm tra độc lập bằng gradient-alignment; gọi port hiện có trên MNIST 4/9 |
| MWN | xjtushujun/Meta-weight-net_class-imbalance | So một bước với MetaModule.update_params nguyên bản; gọi port hiện có trên CIFAR-10 |

## Các khác biệt được công khai

- LRE/CB gốc dùng TensorFlow; code đang dùng là chuyển đổi sang PyTorch, không chạy TensorFlow gốc.
- MWN kiểm tra số học dùng module MetaModule nguyên bản; vòng train dùng port functional_call.
- Hai dataset giữ nhãn gốc. Meta/validation được lấy riêng từ original train, không lấy từ original test.
- MNIST chỉ dùng 4 và 9; CIFAR-10 dùng 10 lớp. Tập train long-tail có lớp lớn nhất 1.000 và tỷ lệ 100.
- SmallCNN (16/32 kênh, FC64), 10 epoch, LR0.03, batch64 không tương đương kiến trúc/ngân sách toàn bộ paper.
- Chọn checkpoint theo accuracy validation không trọng số; các phương pháp dùng chung split và ba seed.
- Chỉ có một recipe cố định, không tìm siêu tham số trên test. Không đánh dấu kiểm chứng đạt chỉ vì phương pháp thắng baseline.

## Những kết luận được hỗ trợ

Kiểm tra số học có thể phát hiện sai công thức hoặc sai bước meta-gradient đã chọn.
Chạy dữ liệu gốc cho biết hành vi của bản port ở thiết lập nhỏ đã mô tả, bao gồm trường hợp học yếu/không cải thiện.
Các kết quả này không xác nhận toàn bộ mã, không chứng minh tái lập điểm số paper, và không biến kết quả âm trên BAF thành bằng chứng cơ chế thất bại nói chung.

## Meta set BAF hiện có

128 mẫu/lớp lấy từ tháng 5 của train, seed317; loại 256 dòng khỏi core train của tất cả đối chứng.
Nhãn lấy từ bộ BAF, không có xác minh chuyên gia bổ sung. Meta set cân bằng là lựa chọn thiết kế,
không đại diện tỷ lệ fraud tự nhiên và không được gọi là ground truth sạch đã kiểm định độc lập.
