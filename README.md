# Dự báo giá điện

Ứng dụng desktop Python dùng để dự báo mức tiêu thụ điện theo giờ trong tương lai (`kWh`) từ các file CSV xuất ra theo kiểu SCADA.

## Dữ liệu đầu vào

Ứng dụng cần một file CSV điện bắt buộc và một file CSV khách hàng tùy chọn:

- `data_2026.csv`: dữ liệu telemetry SCADA có các cột `time,name,original_value_float`, bao gồm `P`, `PF`, `IAVG` và chỉ số `KWH` lũy kế
- File CSV danh sách khách hàng/khách tham quan: file tùy chọn, có một cột thời gian và một cột số lượng như `visitors`, `guest_count` hoặc `customer_count`

Các timestamp gốc được xem là UTC và được chuyển sang múi giờ `Asia/Ho_Chi_Minh`. Sản lượng điện theo giờ (`kWh`) được tính từ phần chênh lệch của chỉ số `KWH` lũy kế trong `data_2026.csv`.
Timestamp của file khách hàng được xem là giờ địa phương Việt Nam, trừ khi dữ liệu đã có sẵn timezone. Nếu file khách hàng có cột `area` hoặc `meter`, dữ liệu sẽ được gộp theo cấp tương ứng; nếu không, dữ liệu sẽ được áp dụng cho tất cả công tơ theo từng giờ.

## Cài đặt

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell không cho chạy script activate, chạy một lần:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Chạy ứng dụng

Ứng dụng desktop native trên macOS:

```bash
bash scripts/setup_native_macos.sh
source .venv-native/bin/activate
python -m electricity_forecast.app
```

Cách chạy native này sử dụng Homebrew Python 3.10 cùng với `python-tk@3.10`. Python hệ thống tại `/usr/bin/python3` không phù hợp cho ứng dụng này vì binding GUI native bị lỗi trên môi trường macOS hiện tại.

Giao diện web cục bộ dự phòng:

```bash
source .venv/bin/activate
ELECTRICITY_FORECAST_UI=web python -m electricity_forecast.app
```

## Nhiệt độ thời tiết

Tab Forecast sử dụng Open-Meteo để lấy nhiệt độ trung bình theo tháng tại địa điểm và tháng được chọn. Địa điểm mặc định là `Hòn Thơm, Phú Quốc`; có thể chọn một địa điểm preset khác hoặc nhập tên địa điểm / `lat,long`. Cần có kết nối Internet để gọi API này.

## Dự báo

Quá trình huấn luyện sử dụng hồi quy tuyến tính theo từng công tơ, kết hợp các đặc trưng thời gian theo giờ, nhiệt độ mô phỏng/thời tiết, số lượng khách, công suất tác dụng `P`, hệ số công suất `PF`, dòng điện trung bình `IAVG`, điện áp trung bình `VAVG`, cùng các đặc trưng kWh dạng lag/rolling. Sau khi huấn luyện, ứng dụng hiển thị biểu đồ scatter backtest giữa Actual và Predicted, được nhóm theo khu vực, đồng thời có thể xuất file CSV backtest.

## Phát hiện bất thường

Tab Anomaly sử dụng Isolation Forest kết hợp với các quy tắc vận hành điện để đánh dấu các chỉ số theo giờ bất thường từ `data_2026.csv`. Tab này đọc phần chênh lệch kWh cùng với các chỉ số `P`, `Q`, `S`, `PF`, `IA`, `IB`, `IC`, `IAVG`, `%V`, `%A`, `VAVG` và các chỉ số THD nếu có. Kết quả bao gồm điểm bất thường, mức độ nghiêm trọng, loại bất thường và lý do, chẳng hạn như tăng đột biến tiêu thụ, quá tải hệ thống, hệ số công suất thấp, lệch pha, điện áp bất thường, méo hài, thiết bị vận hành bất thường, tiêu thụ ngoài giờ hoặc reset/outlier telemetry.

## Quy trình kiểm tra nhanh bằng CLI

```bash
python -m electricity_forecast.cli \
  --telemetry /Users/macbook/Downloads/data_2026.csv \
  --guests /Users/macbook/Downloads/sunworld_honthom_hourly_jan2026.csv \
  --weather-location "Hòn Thơm, Phú Quốc" \
  --weather-month 2026-01 \
  --horizon 168 \
  --output exports/forecast_168h.csv
```

Chạy phát hiện bất thường bằng CLI:

```bash
python -m electricity_forecast.cli \
  --telemetry /Users/macbook/Downloads/data_2026.csv \
  --detect-anomalies \
  --anomaly-contamination 0.05 \
  --anomaly-output exports/anomalies.csv
```

## Đóng gói ứng dụng macOS

```bash
bash scripts/build_macos_app.sh
```

Với tập dữ liệu lớn, nên để các file CSV bên ngoài app bundle và chọn file từ tab Data.

## Đóng gói ứng dụng Windows native

Hãy build trên máy Windows. PyInstaller không thể tạo file Windows `.exe` từ macOS.

```powershell
scripts\build_windows.ps1
```

Kết quả đầu ra là:

```text
dist\ElectricityForecast\ElectricityForecast.exe
```

Bản build Windows cố ý dùng `onedir` thay vì `onefile` để ứng dụng khởi động nhanh. Khi phân phối, cần gửi toàn bộ thư mục `dist\ElectricityForecast`, không chỉ riêng file `.exe`.
