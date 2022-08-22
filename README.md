# PracticeVibrationAnalysis

本リポジトリでは，クラスタリングの入門として，以下を想定した振動解析の例を示す．
- 複数周波数の波を含む振動のダミーデータの作成
- 振幅平均に基づく波形のk-meansクラスタリング
- パワースペクトルに基づく波形のk-meansクラスタリング
- リアルタイムを模擬したクラスタ予測

ただし，振動のノイズはガウス過程とする．また，k-meansで使用する距離尺度はユークリッド距離のみとする．

## Directory Structure

```

.
│   kmeans_ampmean.py       -> 振幅平均でクラスタリング
│   kmeans_fft.py           -> パワースペクトルでクラスタリング
│   rt_detect_ampmean.py    -> 振幅平均による疑似リアルタイムクラスタ予測
│   rt_detect_fft.py        -> パワースペクトルによる疑似リアルタイムクラスタ予測
├───configs                 -> ダミーデータの設定ファイルを格納
│       example1.json
│       example2.json
├───data                    -> 作成データファイルを格納
├───figures                 -> 作成データの波形画像を格納
├───src
│       visualize_data.py   -> グラフ描画のモジュール
│       __init__.py
└───utils
        make_data.py        -> k-meansモデル学習用のデータを作成
        make_rtdata_1.py    -> 振幅平均による疑似リアルタイムクラスタ予測のためのデータ作成
        make_rtdata_2.py    -> パワースペクトルによる疑似リアルタイムクラスタ予測のためのデータ作成
```

## Requirements

- Python 3
- numpy
- scipy
- scikit-learn
- matplotlib

## Usage

```
python ./utils/make_data ./configs/example1.json
python ./utils/make_data ./configs/example2.json

python kmeans_ampmean.py
python kmeans_fft.py

python rt_detect_ampmean.py
python rt_detect_fft.py
```
