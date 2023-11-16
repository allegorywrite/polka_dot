# polka_dot

```
git clone git@github.com:allegorywrite/polka_dot.git
cd polka_dot
conda env create -f=gym.yml
pip install -e .
cd optimal/lie_control
# ダイナミクスの学習
python scripts/control.py
# 軌道の最適化
python scripts/optim.py --open3d
# シミュレーションの描画 --gui 
# matplotlibで描画 --matplotlib
# フィードバック最適化 --feedback
# 最適化の回数(default:1) --optim_itr 10 
```