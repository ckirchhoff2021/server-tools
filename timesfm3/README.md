# TimesFM3 HTTP Service

将 Google TimesFM 3.0 时间序列预测模型封装为 HTTP 接口服务，支持单变量/多变量预测及分位数输出。

## 文件说明

- `timesfm3_http_server.py` — FastAPI 服务端
- `timesfm3_http_examples.py` — Python 调用示例

## 环境依赖

```bash
pip install timesfm torch fastapi uvicorn numpy pydantic
```

需提前下载 TimesFM 3.0 模型权重到本地。

## 启动服务

```bash
python timesfm3_http_server.py --host 0.0.0.0 --port 8333 --device cpu
```

参数说明：
- `--host`: 监听地址，默认 `0.0.0.0`
- `--port`: 端口，默认 `8333`
- `--device`: `auto` / `cpu` / `cuda`
- `--checkpoint`: 权重目录路径

## API 接口

### GET /health

健康检查，返回模型加载状态、设备信息、Torch/CUDA 版本。

```bash
curl http://127.0.0.1:8333/health
```

### POST /predict

时间序列预测。

```bash
curl -X POST http://127.0.0.1:8333/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "context": [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,
                66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81],
    "horizon": 8,
    "return_quantiles": true,
    "use_symmetric_averaging": false
  }'
```

请求字段：
- `context`：历史数据，单变量 `(T,)` 或多变量 `(V, T)` 数值数组，长度至少 32
- `horizon`：预测未来步数，1~2048
- `return_quantiles`：是否返回分位数预测
- `use_symmetric_averaging`：是否启用对称平均（透传给 TimesFM）

响应示例：

```json
{
  "forecast": [73.76, 73.16, 73.00],
  "forecast_shape": [12],
  "quantiles": [[73.70, 73.71, ...]],
  "quantiles_shape": [12, 9]
}
```

## 输入约束

- `context` 必须是数值数组，支持 1D（单变量）或 2D（多变量）
- 时间序列长度至少 32
- 不支持 NaN/inf
- 建议业务数据提前完成缺失值填充、异常值处理、重采样

## Python 客户端示例

```bash
python timesfm3_http_examples.py --base-url http://127.0.0.1:8333
```

会依次验证健康检查、单变量预测、多变量预测。
