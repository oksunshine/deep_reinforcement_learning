これまで表形式表現で行ってきたが、状態変数の量が多くなると表の行数が多くなることが問題点として挙げられる。
これをディープラーニングで解決する。
深層強化学習DQNでは、行動価値関数をディープニューラルネットで表現する。

CartPoleでいえば、入力素子は位置、速度、角度、角速度の４変数であるのに対して、
出力は右と左の２種類である。

出力する値は行動価値関数である。
つまり、その素子に対応する行動を採用した場合、その後に得られるであろう割引報酬和を出力する。
そして、各素子が出力する割引報酬和を比較して行動を決定する。
つまり、回帰問題を解く。

最適な行動価値関数は
ある状態StからAtを選択した時の行動価値は
次に得られる報酬と次の状態から最適な行動を選択した時の行動価値に割引率をかけたものである。

よって理想状態と現在の報酬との二乗誤差を誤差関数として定義する。

４つの工夫点
- experience replay
- fixed target Q-network
- reward clipping
    - aa 
- Huber関数
    - 誤差関数を二乗誤差ではなく、Huber関数を利用した。