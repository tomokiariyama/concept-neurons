# concept neurons
まもなく公開いたします。

・まず，"set_dataset_path.sh"内に，データセットのダウンロード先ディレクトリを指定してください
・次に，"concept_neurons"ディレクトリにいる状態でご自身が使用される仮想環境にスイッチし，"zsh Setup.sh"と実行してください
・次に，"zsh ConceptNet.sh"と実行してください．著者の環境では実行に55分ほどかかりました
・最後に，"zsh make_graphs.sh"と実行してください
・"concept_neurons/figure/ConceptNet/subject/*"配下に，論文と同様のグラフがそれぞれ作成されます
  ・図3: "all_suppressed_graph.png"
  ・図4: "all_enhanced_graph.png"
  ・図5: "all_suppressed_overlapping_histogram.png"
  ・図6: "all_enhanced_overlapping_histogram.png"
  ・図7: "relevant_suppressed_overlapping_histogram.png"
  ・図8: "relevant_enhanced_overlapping_histogram.png"

## スクリプト

```yaml
set_dataset_path.sh: データセットのダウンロード先ディレクトリを指定する（初めに必ず設定してください）
Setup.sh: データセットのダウンロードや，必要なモジュールのインストール
ConceptNet.sh: 実験結果の取得
  evaluate.py: 実験コード
make_graphs.sh: 実験結果のグラフ化
  make_graphs.py: グラフ化コード
```
