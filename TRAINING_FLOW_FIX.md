# 訓練等待與權重續訓流程修正

本次修正位於 `模型優化` 分支，針對 `train.py` 的兩個問題：資料不足時永遠等待，以及已載入的權重被第二次模型初始化覆蓋。新增 `test_training_flow.py` 驗證流程。

## 一、資料不足時為什麼會卡住

修改前：

```python
while len(self.data_buffer) <= self.batch_size:
    time.sleep(600)
```

這裡的 `self.data_buffer` 是先前讀進記憶體的資料。收集程式更新磁碟檔案，不會自動改變這個 Python 物件；因此睡醒後仍然看到相同長度。另外，原本使用 `<=`，剛好有 512 筆、batch size 也是 512 時也不能訓練。

修改後分成三層：

1. `_read_training_data()` 每次重新讀取檔案或 Redis，回傳資料和 `iters` 版本。
2. `_wait_for_training_data(last_iters)` 確認樣本數至少等於 batch size，且這個版本尚未在本次執行中訓練。未達條件時等待，再重新讀取。
3. `run()` 在資料準備好後執行一次更新，完成後才記錄已訓練的版本。

```python
generation = self._wait_for_training_data(last_iters)
loss, entropy = self.policy_updata()
last_iters = generation
```

例如來源第一次只有 300 筆，等待後增加至 512 筆，下一次讀取便會開始訓練。資料不足的讀取不會消耗版本或訓練批次；來源尚未產生新版本時，也不會一直重複訓練同一版本。

檔案不存在、暫時寫入不完整或缺少資料欄位時，會顯示原因並重試。`KeyboardInterrupt` 仍可中止等待。預設檔案模式每 30 秒重讀、Redis 每 5 秒重讀；可在 `CONFIG` 設定正數 `train_data_poll_interval` 調整間隔。這不同於 `train_update_interval`：後者是一次訓練更新完成後的等待時間。

## 二、續訓為什麼會被重新初始化

修改前 `TrainPipeline.__init__()` 有兩段模型初始化：前段載入 checkpoint，後段又依 `init_model` 建立模型。未指定 `init_model` 時，後段執行 `PolicyValueNet()`，會把剛載入的權重換成隨機權重，但 `start_iter` 還是 checkpoint 的批次編號。

此外，原本 `start_iter` 一律取自自動找到的 checkpoint，即使實際指定載入另一份權重，也可能使用不相符的批次編號；腳本入口固定傳入 current 權重，會遮蔽自動找 checkpoint 的功能。

修改後由 `_select_model()` 決定唯一來源，再呼叫一次 `PolicyValueNet`：

| 順序 | 來源 | 訓練起點 |
| --- | --- | --- |
| 1 | 明確傳入的 `init_model` | 若檔名符合 checkpoint 格式，使用該檔批次；其他權重從 0 計數 |
| 2 | `MODELS_DIR` 中批次數最大的合法 checkpoint | 使用該檔批次 |
| 3 | `CURRENT_POLICY_PATH` | 沒有批次資訊，從 0 計數，但保留載入的權重 |
| 4 | 沒有任何上述檔案 | 建立新模型，從 0 計數 |

合法 checkpoint 名稱為 `new_current_policy_batch數字.pth`，例如 `new_current_policy_batch200.pth`。選擇時比較數字批次，而非檔案修改時間，避免複製舊檔後反而將它視為最新。暗棋／全明棋仍使用原本設定對應的模型目錄。

`_checkpoint_step()` 只解析實際選中的檔案，不再拿另一個 checkpoint 的批次數。`loaded_model_path` 記錄本次選中的來源；載入 batch 200 的 checkpoint，下一次更新顯示為第 201 批。

指定檔案不存在時拋出 `FileNotFoundError`；選中的檔案損壞或網路結構不相容時，保留載入錯誤並停止。這避免錯誤被當作「可以從隨機模型繼續訓練」。

## 三、各修改位置與理由

| 位置 | 修改與理由 |
| --- | --- |
| `TrainPipeline.__init__()` | 合併模型初始化、記錄唯一載入來源與正確批次；設定資料輪詢間隔，檢查 buffer 容量能否容納一個 batch。 |
| `_select_model()` | 集中管理明確的來源優先順序；排除不合法 checkpoint 檔名。 |
| `_checkpoint_step()` | 避免模型權重與批次編號來自不同檔案。 |
| `_read_training_data()` | 每輪重新載入資料；Redis 改為重讀快照並取代記憶體資料，不依舊長度推算讀取位置，也不由訓練端刪除 Redis 項目。若讀取前後版本不同則重試。 |
| `_wait_for_training_data()` | 統一處理資料不足、來源暫不可讀及等待新版本；接受 `len(data) == batch_size`。 |
| `run()` | 移除只睡眠而不重讀的內層迴圈；真正完成更新才消耗資料版本。 |
| 程式入口 | 加入 `if __name__ == '__main__'`，避免匯入模組就啟動長時間訓練；直接執行時使用自動選擇模型流程。 |
| 目錄建立及 RandomPlayer 匯入 | 建立模型目錄移至建構函數，RandomPlayer 延後至評估函數才匯入。測試／工具只匯入 `train` 時不啟動訓練、不建立模型目錄，也不需要先載入評估玩家的額外依賴。 |
| `test_training_flow.py` | 使用替身模型、暫存資料與模擬等待，驗證實際函數的載入次數、資料重讀及完整 `run()` 接線。 |

## 四、使用方式

在專案資料夾執行：

```powershell
python train.py
```

自動依「checkpoint → current 權重 → 新模型」選擇。明確指定某份權重時，可在 Python 使用：

```python
from train import TrainPipeline

pipeline = TrainPipeline(init_model='new_models/new_current_policy_batch200.pth')
pipeline.run()
```

上述相對路徑以啟動時的工作目錄為準，因此應從專案資料夾執行。終端會列出實際載入檔案與批次；失敗時請處理檔案／結構問題後再啟動。

## 五、本次續訓的範圍

**本次保證既有權重不被重新初始化，屬於權重續訓，不是完整恢復訓練器狀態。** 原有 `.pth` 仍只保存模型權重，optimizer 的動量、學習率調整倍率、隨機狀態及上次消耗的資料版本未保存；重新啟動時這些狀態會重新建立。一般 current 權重檔沒有批次資訊，不能憑空恢復其訓練步數。

本次未修改網路結構、value 標籤、吃子偏好、終局規則、收集端 buffer 重複累加問題或最佳模型的評估／保存門檻。資料 writer 的原子寫入與 Redis 的交易一致性仍屬後續工作；目前讀取重試與版本前後檢查並不等於完整交易快照。

## 六、驗證方式

```powershell
python -B -m unittest test_training_flow test_chance_mcts -v
```

新增的 17 個訓練流程測試包含：自動／指定模型只載入一次、checkpoint 批次選擇、current 回退、從零建立、缺檔／壞檔不偷換模型、資料不足後重讀、剛好一個 batch、已訓練版本等待、損壞資料重試、可中止等待、兩批更新接線、Redis 快照替換與版本變動，以及單純匯入不啟動訓練。它們不會進行長時間模型訓練，也不改寫專案原有權重和資料。

既有 `test_terminal_rules.py` 的缺少 `reset_terminal_history()` 問題與本次修正無關；本次測試指令不包含該尚未完成的終局規則套件。


## 七、本次驗證結果

- 在實際「模型優化」分支執行：17 個新增訓練流程測試與 20 個既有翻棋測試，共 37 個全部通過。
- 以 CPU 載入實際 `new_current_policy.pth`：確認模型建構僅一次，全部載入參數與原權重檔逐一相等，沒有被重新隨機初始化。
- 真實模型驗證沒有執行梯度更新或保存模型；Redis 路徑使用替身服務測試，未連接實際 Redis。
- Git 差異檢查通過。這是功能回歸驗證，不代表已完成長時間訓練或棋力評估。
