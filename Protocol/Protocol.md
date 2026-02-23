# PMU Edge AI CSV Protocol v1.2
# PMU 边缘智能 CSV 数据协议规范 v1.2

---

## 1. Purpose | 协议目的

**English**

This document defines a standardized CSV protocol for PMU-based edge AI systems.  
It ensures consistency for:

- Fault classification
- Disturbance detection
- Raspberry Pi deployment
- Jetson Nano deployment
- Cross-platform model reproducibility

Each row represents one synchronized PMU measurement sample.

**中文**

本协议定义了用于 PMU 边缘智能系统的标准 CSV 数据格式。  
适用于：

- 电力故障分类
- 动态扰动检测
- 树莓派部署
- Jetson Nano 部署
- 跨平台模型复现

每一行代表一个同步 PMU 采样点。

---

# 2. Strict Column Order | 严格字段顺序

| Index | Field      | Description                 | Unit |
| ----: | ---------- | --------------------------- | ---- |
|     1 | DFDT       | Rate of Change of Frequency | Hz/s |
|     2 | FREQ       | System Frequency            | Hz   |
|     3 | IA_MAG     | Phase A Current Magnitude   | A    |
|     4 | IA_ANG     | Phase A Current Angle       | deg  |
|     5 | IB_MAG     | Phase B Current Magnitude   | A    |
|     6 | IB_ANG     | Phase B Current Angle       | deg  |
|     7 | IC_MAG     | Phase C Current Magnitude   | A    |
|     8 | IC_ANG     | Phase C Current Angle       | deg  |
|     9 | VA_MAG     | Phase A Voltage Magnitude   | V    |
|    10 | VA_ANG     | Phase A Voltage Angle       | deg  |
|    11 | VB_MAG     | Phase B Voltage Magnitude   | V    |
|    12 | VB_ANG     | Phase B Voltage Angle       | deg  |
|    13 | VC_MAG     | Phase C Voltage Magnitude   | V    |
|    14 | VC_ANG     | Phase C Voltage Angle       | deg  |
|    15 | TIMESTAMP  | Time string (MM:SS.s)       | -    |
|    16 | ERROR_CODE | Encoded fault code          | int  |

---

# 3. Timestamp Format | 时间格式

Format:
```
MM:SS.s
```

Examples:
```
00:00.0
00:00.1
01:15.3
```

Rules:
- Must be monotonic increasing
- Must be parseable to seconds

中文说明：
- 时间必须单调递增
- 必须可转换为秒

---

# 4. ERROR_CODE Encoding Scheme
# 错误码编码规则

---

## 4.1 Encoding Formula | 编码公式

```
ERROR_CODE = S * 100 + E
```

Where:

- S = Severity Level (hundreds digit)
- E = Error ID (00–99)

So:

- Hundreds digit → Severity
- Last two digits → Specific error

Example:

- 000 → Normal
- 201 → Critical SLG fault
- 150 → Warning data missing

---

## 4.2 Severity Levels | 严重等级

| S    | Level    | Meaning    | Edge Action |
| ---- | -------- | ---------- | ----------- |
| 0    | NORMAL   | 正常       | log         |
| 1    | WARNING  | 轻微异常   | warn        |
| 2    | CRITICAL | 严重故障   | alarm       |
| 3    | FATAL    | 不可用数据 | drop        |

Parsing:

```
S = ERROR_CODE // 100
E = ERROR_CODE % 100
```

Domain Rule:

- E < 50 → Power System Fault
- E ≥ 50 → Data Integrity Error

---

# 5. Error Definitions | 错误码定义

---

## 5.1 Power System Faults (E = 00–49)

| ERROR_CODE | Name              | Severity | Description                 |
| ---------- | ----------------- | -------- | --------------------------- |
| 000        | NORMAL            | 0        | Normal operation            |
| 201        | SLG_FAULT         | 2        | Single line-to-ground fault |
| 202        | LL_FAULT          | 2        | Line-to-line fault          |
| 203        | DLG_FAULT         | 2        | Double line-to-ground fault |
| 204        | THREE_PHASE_FAULT | 2        | Three-phase short circuit   |
| 105        | VOLTAGE_SAG       | 1        | Voltage drop                |
| 106        | VOLTAGE_SWELL     | 1        | Voltage rise                |
| 107        | FREQ_DEVIATION    | 1        | Frequency deviation         |
| 108        | OSCILLATION       | 1        | Low frequency oscillation   |
| 109        | UNBALANCED        | 1        | Phase imbalance             |

中文说明：

- 2xx → 严重电力故障
- 1xx → 警告级别电力异常

---

## 5.2 Data Integrity Errors (E = 50–99)

| ERROR_CODE | Name                 | Severity | Description              |
| ---------- | -------------------- | -------- | ------------------------ |
| 150        | DATA_MISSING         | 1        | Missing value            |
| 151        | FLAG_ERROR           | 1        | PMU flag invalid         |
| 252        | TIME_SYNC_ERROR      | 2        | Timestamp discontinuity  |
| 153        | MAG_OUT_OF_RANGE     | 1        | Magnitude out of range   |
| 154        | ANGLE_DISCONTINUITY  | 1        | Angle jump error         |
| 155        | DFDT_OUTLIER         | 1        | DFDT abnormal            |
| 256        | FREQ_SENSOR_FAULT    | 2        | Frequency sensor failure |
| 257        | CURRENT_SENSOR_FAULT | 2        | Current channel fault    |
| 258        | VOLT_SENSOR_FAULT    | 2        | Voltage channel fault    |
| 359        | UNUSABLE_SAMPLE      | 3        | Sample invalid           |

---

# 6. CSV Header Template

```csv
DFDT,FREQ,IA_MAG,IA_ANG,IB_MAG,IB_ANG,IC_MAG,IC_ANG,VA_MAG,VA_ANG,VB_MAG,VB_ANG,VC_MAG,VC_ANG,TIMESTAMP,ERROR_CODE
```

---

# 7. Example Rows

Normal:

```csv
0.00012,49.98,102.15,-150.32,105.92,-30.51,98.43,89.78,241.65,-150.51,242.90,-30.61,243.14,89.79,00:00.1,000
```

Critical 3-phase fault:

```csv
0.50000,48.90,310.20,-140.00,305.10,-20.00,299.30,100.00,120.00,-160.00,118.00,-40.00,121.00,80.00,00:02.4,204
```

Warning data missing:

```csv
0.00010,50.00,NaN,NaN,105.00,-30.50,98.00,90.00,241.00,-150.00,243.00,-30.60,243.00,89.80,00:03.1,150
```

---

# 8. Modeling Recommendations | 建模建议

English:

- Convert (MAG, ANG) → (Re, Im)
- Normalize DFDT and FREQ
- Drop samples where S ≥ 3

中文：

- 建议将幅值角度转换为实虚部
- 标准化 DFDT 和 FREQ
- S ≥ 3 时丢弃样本

---

# 10. Missing Data Handling Specification  
# 10. 缺失数据处理规范

## 10.1 General Rule | 总体规则

**English**

If any required field is unavailable, missing, or cannot be parsed,
the value MUST be recorded as:

```
NaN
```

The string must follow IEEE floating-point representation
and must not be replaced by:

- 0
- empty string
- -1
- arbitrary placeholders

**中文**

如果任意字段数据缺失、不可解析或无法计算，
必须使用：

```
NaN
```

表示缺失值。

禁止使用以下替代方式：

- 0
- 空字符串
- -1
- 任意自定义占位符

---

## 10.2 Affected Fields | 适用字段

The NaN rule applies to all numeric columns:

- DFDT
- FREQ
- IA_MAG, IA_ANG
- IB_MAG, IB_ANG
- IC_MAG, IC_ANG
- VA_MAG, VA_ANG
- VB_MAG, VB_ANG
- VC_MAG, VC_ANG

TIMESTAMP must not be NaN. If timestamp is invalid:

- ERROR_CODE must be set to 252 (TIME_SYNC_ERROR)

---

## 10.3 ERROR_CODE Interaction | 错误码联动规则

If a row contains one or more NaN values:

### Case 1 — Base state is NORMAL (000)
ERROR_CODE must be updated to:

```
150
```

(DATA_MISSING — Warning Level)

### Case 2 — Base state is Power Fault (2xx or 1xx)
NaN values must still be recorded,
but ERROR_CODE should remain unchanged unless:

- more than 50% of fields are NaN → set to 359 (UNUSABLE_SAMPLE)

---

## 10.4 Edge Deployment Behavior | 边缘设备处理建议

When deployed:

- Rows with ERROR_CODE ≥ 300 must not be used for inference.
- Rows with 150 (DATA_MISSING) may be:
  - interpolated
  - skipped
  - or passed with masking (model dependent)

---

## 10.5 CSV Example with NaN | 示例

```csv
0.00012,49.98,NaN,NaN,105.92,-30.51,98.43,89.78,241.65,-150.51,242.90,-30.61,243.14,89.79,00:00.1,150
```

Meaning:

- IA data missing
- System is still running
- Warning level triggered

---

## 10.6 Rationale | 设计原因

Using NaN ensures:

- Compatibility with NumPy / Pandas
- Compatibility with PyTorch / TensorFlow
- Proper masking capability
- Clear separation between missing data and true zero values

This avoids:

- False physical interpretation
- Silent corruption
- Training bias

---

