import copy
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_LEN = 90
TARGET_COL = "Global_active_power"
DROP_COLS = {"datetime", "NBJBROU"}
SEEDS = [42, 123, 2024, 7, 99]

MAX_EPOCHS = 120
PATIENCE = 12
BATCH_SIZE = 32
LR = 5e-4
MAX_FOLDS = 3
VAL_EXTRA_DAYS = 89

D_MODEL = 64
CNN_OUT_CHANNELS = 64
CNN_KERNEL_SIZE = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionCNN(nn.Module):
    """
    带注意力机制的CNN：先用CNN提取局部特征，再用注意力机制自适应加权聚合
    相比平均池化，注意力机制让模型自动学习哪些时间步对预测更重要
    """
    def __init__(self, input_size: int, output_channels: int = 64, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, output_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(output_channels, output_channels // 4),
            nn.Tanh(),
            nn.Linear(output_channels // 4, 1)
        )
        self.proj = nn.Linear(output_channels, output_channels)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        conv_out = self.conv(x)
        conv_out = conv_out.permute(0, 2, 1)
        attn_weights = self.attention(conv_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted = (conv_out * attn_weights).sum(dim=1)
        return self.proj(weighted), attn_weights.squeeze(-1)


class AttnCNNTransformer(nn.Module):
    """
    注意力CNN + Transformer（不含周期感知嵌入）
    """
    def __init__(
        self,
        feature_size: int,
        d_model: int = D_MODEL,
        nhead: int = 4,
        num_layers: int = 2,
        output_len: int = 90,
        dropout: float = 0.1,
        cnn_out_channels: int = CNN_OUT_CHANNELS,
        kernel_size: int = CNN_KERNEL_SIZE,
    ):
        super().__init__()
        self.cnn = AttentionCNN(feature_size, cnn_out_channels, kernel_size)
        self.cnn_proj = nn.Linear(cnn_out_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_len)

    def forward(self, x: torch.Tensor):
        B, seq_len, _ = x.shape

        cnn_feat, attn_weights = self.cnn(x)
        cnn_feat = self.cnn_proj(cnn_feat)
        cnn_feat = cnn_feat.unsqueeze(1).expand(-1, seq_len, -1)
        cnn_feat = self.pos_encoder(cnn_feat)
        cnn_feat = self.norm(cnn_feat)
        cnn_feat = self.dropout(cnn_feat)

        encoded = self.transformer_encoder(cnn_feat)
        last_state = encoded[:, -1, :]
        return self.head(last_state), attn_weights


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in DROP_COLS]


def build_windows(
    feat_array: np.ndarray,
    target_array: np.ndarray,
    input_len: int,
    output_len: int,
    start_idx: int = 0,
    end_idx: int | None = None,
):
    last_start = len(feat_array) - input_len - output_len
    if end_idx is None:
        end_idx = last_start
    end_idx = min(end_idx, last_start)

    xs, ys = [], []
    for start in range(start_idx, end_idx + 1):
        xs.append(feat_array[start : start + input_len])
        ys.append(target_array[start + input_len : start + input_len + output_len])

    if not xs:
        return (
            np.empty((0, input_len, feat_array.shape[1]), dtype=np.float32),
            np.empty((0, output_len), dtype=np.float32),
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def build_rolling_folds(
    train_df: pd.DataFrame,
    output_len: int,
    input_len: int,
    max_folds: int = MAX_FOLDS,
    val_extra_days: int = VAL_EXTRA_DAYS,
):
    total_len = len(train_df)
    min_train_days = input_len + output_len
    max_val_days = total_len - min_train_days
    if max_val_days < output_len:
        return []

    val_days = min(output_len + val_extra_days, max_val_days)
    folds = []
    val_end = total_len

    while len(folds) < max_folds:
        val_start = val_end - val_days
        train_end = val_start
        if train_end < min_train_days:
            break
        folds.append({"train_end": train_end, "val_start": val_start, "val_end": val_end})
        val_end = val_start

    folds.reverse()
    return folds


def fit_scalers(train_df: pd.DataFrame, feat_cols: list[str]):
    feat_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feat_scaler.fit(train_df[feat_cols].values)
    target_scaler.fit(train_df[[TARGET_COL]].values)
    return feat_scaler, target_scaler


def prepare_train_dataset(
    train_df: pd.DataFrame,
    feat_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feat_cols: list[str],
    output_len: int,
):
    feat = feat_scaler.transform(train_df[feat_cols].values)
    target = target_scaler.transform(train_df[[TARGET_COL]].values).flatten()
    x, y = build_windows(feat, target, INPUT_LEN, output_len)
    return SequenceDataset(x, y)


def prepare_eval_dataset(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feat_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feat_cols: list[str],
    output_len: int,
):
    history_tail = history_df.iloc[-INPUT_LEN:].copy()
    full_df = pd.concat([history_tail, eval_df], ignore_index=True)
    feat = feat_scaler.transform(full_df[feat_cols].values)
    target = target_scaler.transform(full_df[[TARGET_COL]].values).flatten()
    x, y = build_windows(
        feat,
        target,
        INPUT_LEN,
        output_len,
        start_idx=0,
        end_idx=len(eval_df) - output_len,
    )
    return SequenceDataset(x, y)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred, _ = model(x)
            total_loss += criterion(pred, y).item() * x.size(0)
    return total_loss / len(loader.dataset)


def run_training(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset | None,
    output_len: int,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
):
    set_seed(seed)

    model = AttnCNNTransformer(
        feature_size=train_dataset.features.shape[-1],
        d_model=D_MODEL,
        nhead=4,
        num_layers=2,
        output_len=output_len,
        dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_score = float("inf")
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

        if val_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_score = train_loss
            continue

        val_loss = evaluate_loss(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_score:
            best_score = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_score


def inverse_metrics(
    model: nn.Module,
    dataset: SequenceDataset,
    target_scaler: MinMaxScaler,
):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred, _ = model(x.to(DEVICE))
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())

    pred_arr = np.concatenate(preds, axis=0)
    true_arr = np.concatenate(trues, axis=0)
    pred_inv = target_scaler.inverse_transform(pred_arr.reshape(-1, 1)).reshape(pred_arr.shape)
    true_inv = target_scaler.inverse_transform(true_arr.reshape(-1, 1)).reshape(true_arr.shape)

    window_mse = np.mean((pred_inv - true_inv) ** 2, axis=1)
    window_mae = np.mean(np.abs(pred_inv - true_inv), axis=1)
    mse = float(np.mean(window_mse))
    mae = float(np.mean(window_mae))
    return mse, mae, pred_inv, true_inv, window_mse, window_mae


def select_epochs_by_rolling_validation(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    output_len: int,
    seed: int,
):
    folds = build_rolling_folds(train_df, output_len, INPUT_LEN)
    if not folds:
        raise ValueError(
            f"train.csv is too short for rolling validation with output_len={output_len}."
        )

    best_epochs = []
    fold_scores = []

    for fold_id, fold in enumerate(folds, start=1):
        train_fold = train_df.iloc[: fold["train_end"]].reset_index(drop=True)
        val_fold = train_df.iloc[fold["val_start"] : fold["val_end"]].reset_index(drop=True)

        feat_scaler, target_scaler = fit_scalers(train_fold, feat_cols)
        train_dataset = prepare_train_dataset(
            train_fold, feat_scaler, target_scaler, feat_cols, output_len
        )
        val_dataset = prepare_eval_dataset(
            train_fold, val_fold, feat_scaler, target_scaler, feat_cols, output_len
        )

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            continue

        _, best_epoch, best_score = run_training(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_len=output_len,
            seed=seed,
        )
        best_epochs.append(best_epoch)
        fold_scores.append(best_score)
        print(
            f"    Fold {fold_id}: train_days={len(train_fold)}, "
            f"val_days={len(val_fold)}, best_epoch={best_epoch}, val_loss={best_score:.5f}"
        )

    if not best_epochs:
        raise ValueError(
            f"Could not create any valid rolling-validation folds for output_len={output_len}."
        )

    selected_epoch = int(round(float(np.median(best_epochs))))
    mean_score = float(np.mean(fold_scores))
    return selected_epoch, folds, mean_score


def train_final_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    output_len: int,
    seed: int,
):
    if len(test_df) < output_len:
        raise ValueError(
            f"test.csv has {len(test_df)} days, which is not enough for {output_len}-day forecasting."
        )

    selected_epoch, folds, cv_score = select_epochs_by_rolling_validation(
        train_df, feat_cols, output_len, seed
    )
    feat_scaler, target_scaler = fit_scalers(train_df, feat_cols)

    train_dataset = prepare_train_dataset(
        train_df, feat_scaler, target_scaler, feat_cols, output_len
    )
    test_dataset = prepare_eval_dataset(
        train_df, test_df, feat_scaler, target_scaler, feat_cols, output_len
    )

    final_model, _, _ = run_training(
        train_dataset=train_dataset,
        val_dataset=None,
        output_len=output_len,
        seed=seed,
        max_epochs=selected_epoch,
        patience=PATIENCE,
    )

    mse, mae, preds, trues, window_mse, window_mae = inverse_metrics(
        final_model, test_dataset, target_scaler
    )
    return {
        "mse": mse,
        "mae": mae,
        "preds": preds,
        "trues": trues,
        "window_mse": window_mse,
        "window_mae": window_mae,
        "selected_epoch": selected_epoch,
        "num_folds": len(folds),
        "cv_score": cv_score,
        "num_test_samples": len(test_dataset),
    }


def aggregate_test_predictions(pred_windows: np.ndarray, test_len: int) -> np.ndarray:
    horizon = pred_windows.shape[1]
    sums = np.zeros(test_len, dtype=np.float64)
    counts = np.zeros(test_len, dtype=np.float64)

    for start_idx in range(len(pred_windows)):
        end_idx = min(start_idx + horizon, test_len)
        width = end_idx - start_idx
        sums[start_idx:end_idx] += pred_windows[start_idx, :width]
        counts[start_idx:end_idx] += 1

    if np.any(counts == 0):
        raise ValueError("Some test days were not covered by rolling predictions.")

    return sums / counts


def save_prediction_plot(results: dict[int, dict], output_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, horizon in zip(axes, [90, 365]):
        if horizon not in results:
            ax.set_visible(False)
            continue

        pred_curve = results[horizon]["aggregated_pred"]
        true_curve = results[horizon]["test_truth"]
        ax.plot(true_curve, label="Ground Truth", color="steelblue")
        ax.plot(pred_curve, label="AttnCNN+Transformer", color="tomato", alpha=0.85)
        ax.set_title(f"AttnCNN+Transformer Rolling Forecast (horizon = {horizon} days)")
        ax.set_xlabel("Test day index")
        ax.set_ylabel("Global_active_power")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    print(f"Using device: {DEVICE}")

    train_df = load_dataframe(TRAIN_PATH)
    test_df = load_dataframe(TEST_PATH)

    feat_cols = get_feature_columns(train_df)
    print(
        f"Loaded train.csv: {len(train_df)} rows, "
        f"{train_df['datetime'].iloc[0].date()} -> {train_df['datetime'].iloc[-1].date()}"
    )
    print(
        f"Loaded test.csv: {len(test_df)} rows, "
        f"{test_df['datetime'].iloc[0].date()} -> {test_df['datetime'].iloc[-1].date()}"
    )
    print(f"Features: {feat_cols}")
    print("Dropped constant/weather column from features: ['NBJBROU']")

    results = {}

    for horizon in [90, 365]:
        print("\n" + "=" * 70)
        print(f"AttnCNN+Transformer direct multi-step forecasting: output_len={horizon}")
        print("=" * 70)

        if len(test_df) < horizon:
            print(
                f"Skip {horizon}-day task: test.csv only has {len(test_df)} days. "
                f"Please reserve at least {horizon} test days."
            )
            continue

        mse_list, mae_list = [], []
        horizon_result = None

        for run_id, seed in enumerate(SEEDS, start=1):
            print(f"\n  Run {run_id}/{len(SEEDS)} | seed={seed}")
            result = train_final_and_evaluate(
                train_df=train_df,
                test_df=test_df,
                feat_cols=feat_cols,
                output_len=horizon,
                seed=seed,
            )
            mse_list.append(result["mse"])
            mae_list.append(result["mae"])
            horizon_result = result
            print(
                f"    selected_epoch={result['selected_epoch']}, "
                f"rolling_folds={result['num_folds']}, "
                f"cv_loss={result['cv_score']:.5f}, "
                f"test_samples={result['num_test_samples']}"
            )
            print(f"    Test MSE={result['mse']:.4f}, Test MAE={result['mae']:.4f}")

        results[horizon] = {
            "mse_list": mse_list,
            "mae_list": mae_list,
            "preds": horizon_result["preds"],
            "trues": horizon_result["trues"],
            "window_mse": horizon_result["window_mse"],
            "window_mae": horizon_result["window_mae"],
            "aggregated_pred": aggregate_test_predictions(
                horizon_result["preds"], len(test_df)
            ),
            "test_truth": test_df[TARGET_COL].to_numpy(),
        }

        print(
            f"\n  Summary ({horizon}d): "
            f"MSE={np.mean(mse_list):.4f} +- {np.std(mse_list):.4f}, "
            f"MAE={np.mean(mae_list):.4f} +- {np.std(mae_list):.4f}"
        )

    if results:
        print("\n" + "=" * 70)
        print("Final summary")
        print("=" * 70)
        summary_rows = []
        for horizon, result in results.items():
            summary_rows.append(
                {
                    "Model": "AttnCNN+Transformer",
                    "Task": f"{horizon}-day forecast",
                    "MSE mean": np.mean(result["mse_list"]),
                    "MSE std": np.std(result["mse_list"]),
                    "MAE mean": np.mean(result["mae_list"]),
                    "MAE std": np.std(result["mae_list"]),
                }
            )
        print(pd.DataFrame(summary_rows).to_string(index=False))

        plot_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "attncnn_transformer.png"
        )
        save_prediction_plot(results, plot_path)
        print(f"\nSaved prediction plot to: {plot_path}")


if __name__ == "__main__":
    main()