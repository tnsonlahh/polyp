# SSSS


2. Cài đặt dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

   Nếu dùng virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

## 2. Chuẩn bị dữ liệu Kvasir

1. Tải dataset Kvasir-SEG raw về máy.
2. Đặt thư mục raw sao cho có cấu trúc:
   ```text
   kvasir-seg/Kvasir-SEG/images/
   kvasir-seg/Kvasir-SEG/masks/
   ```

3. Chạy script chuẩn hóa dữ liệu:
   ```bash
   python3 prepare_kvasir.py --src kvasir-seg/Kvasir-SEG --out data_processed/kvasir
   ```

   - `--src` là đường dẫn đến dataset raw Kvasir.
   - `--out` là thư mục đầu ra chứa dữ liệu đã chuyển sang `.npy`.
   - Mặc định sẽ tạo:
     - `data_processed/kvasir/images/*.npy`
     - `data_processed/kvasir/labels/*.npy`
     - `data_processed/kvasir/fold_label_1.txt` ... `fold_label_5.txt`
     - `data_processed/kvasir/unlabeled.txt`

4. Nếu cần thay số fold hoặc seed:
   ```bash
   python3 prepare_kvasir.py --src kvasir-seg/Kvasir-SEG --out data_processed/kvasir --folds 5 --seed 1
   ```

## 3. Chạy training

Sử dụng file cấu hình `Configs/kvasir_seg.yml` để train theo Kvasir.

Ví dụ với script `train_sup.py`:
```bash
python3 train_sup.py --config_yml Configs/kvasir_seg.yml --gpu 0
```

Các script training khác cũng chạy tương tự, chỉ thay tên file:
```bash
python3 fixmatch.py --config_yml Configs/kvasir_seg.yml --gpu 0
python3 mean_teacher.py --config_yml Configs/kvasir_seg.yml --gpu 0
python3 cps.py --config_yml Configs/kvasir_seg.yml --gpu 0
python3 ccvc.py --config_yml Configs/kvasir_seg.yml --gpu 0
python3 corrmatch.py --config_yml Configs/kvasir_seg.yml --gpu 0
python3 GTA_seg.py --config_yml Configs/kvasir_seg.yml --gpu 0
python3 DMT.py --config_yml Configs/kvasir_seg.yml --gpu 0
```

Nếu script yêu cầu thêm tham số `--exp` hoặc `--exp_name`, bạn có thể thêm như:
```bash
python3 train_sup.py --config_yml Configs/kvasir_seg.yml --gpu 0 --exp my_experiment
```

## 4. Lưu ý

- `prepare_kvasir.py` phải chạy trước để tạo dữ liệu `data_processed/kvasir`.
- Training sử dụng `Configs/kvasir_seg.yml` để lấy cấu hình Kvasir.
- Hiện tại các file train mặc định được thiết lập để labeled data chiếm khoảng 40% tổng dữ liệu train.
- Nếu muốn chỉ dùng 20% labeled, cần sửa lại cấu hình `data.supervised_ratio` trong `Configs/kvasir_seg.yml` và/hoặc chuyển sang sử dụng `get_dataset(... supervised_ratio=0.2 ...)`.
- Nếu raw data ở vị trí khác, sửa lại `--src` cho phù hợp.
- Nếu dùng một GPU khác, thay `--gpu 0` bằng `--gpu 1` hoặc giá trị phù hợp.
