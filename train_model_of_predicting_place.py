import torch
import torch.nn as nn
import os
import logzero
from logzero import logger
from model import model_of_predicting_place
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from tqdm import tqdm
import time


def data_load(minibatch_size):
    # 必要なパスと読み込み
    x_train_save_path_tensor = os.path.join("work", "predicting_places", "x_train_tensor.pt")
    y_train_save_path_tensor = os.path.join("work", "predicting_places", "y_train_tensor.pt")
    x_valid_save_path_tensor = os.path.join("work", "predicting_places", "x_valid_tensor.pt")
    y_valid_save_path_tensor = os.path.join("work", "predicting_places", "y_valid_tensor.pt")
    #x_test_save_path_tensor = os.path.join("work", "predicting_places", "x_test_tensor.pt")
    #y_test_save_path_tensor = os.path.join("work", "predicting_places", "y_test_tensor.pt")

    # 作成した学習・検証データのロード，Q78ではGPUに送る
    X_train = torch.load(x_train_save_path_tensor, map_location=torch.device('cpu'))  # cpuでいいのか？
    Y_train = torch.load(y_train_save_path_tensor, map_location=torch.device('cpu'))
    X_valid = torch.load(x_valid_save_path_tensor, map_location=torch.device('cpu'))
    Y_valid = torch.load(y_valid_save_path_tensor, map_location=torch.device('cpu'))
    #X_test = torch.load(x_test_save_path_tensor, map_location=torch.device('cpu'))
    #Y_test = torch.load(y_test_save_path_tensor, map_location=torch.device('cpu'))

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, Y_valid)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset)
    #test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    #test_data_loader = torch.utils.data.DataLoader(test_dataset)

    return train_data_loader, valid_data_loader, X_train, Y_train, X_valid, Y_valid


def train_loop(dataloader, model, loss_fn, optimizer, device):
    running_loss = 0.0
    model.train()
    for X, label in dataloader:  # ミニバッチサイズごとに処理してくれる
        # 予測と損失の計算
        pred = model(X.to(device))
        loss = loss_fn(pred, label.to(device))

        # バックプロパゲーション
        optimizer.zero_grad()  # モデルパラメータの勾配をリセット
        loss.backward()  # バックプロパゲーションを実行
        optimizer.step()  # 各パラメータの勾配を使用してパラメータの値を調整

        running_loss += loss.item()

    return running_loss / len(dataloader)  # ミニバッチ数に届かない最後の端数のデータセットも計算に含まれているようだ


def get_valid_loss(dataloader, model, loss_fn):
    # 予測と損失の計算
    running_loss = 0.0
    model.eval()
    model.cpu()
    for X, label in dataloader:
        pred = model(X)
        loss = loss_fn(pred, label)

        running_loss += loss.item()

    return running_loss / len(dataloader)


def save_checkpoint(epoch, model, opt, loss, num):
    digit = len(str(num))  # 0埋めするための桁数を全体で回すエポック数からとってくる
    epoch_for_save = str(epoch).zfill(digit)
    os.makedirs(os.path.join("/work02", "ariyama", "exp2022", "concept_neurons", "checkpoints", "predicting_places"), exist_ok=True)
    outfile = os.path.join("/work02", "ariyama", "exp2022", "concept_neurons", "checkpoints", "predicting_places", f'checkpoint_{epoch_for_save}.cpt')
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'train_loss': loss,
                }, outfile)  # pythonのdict型を使って保存
    logger.info(f'successfully saved checkpoint_{epoch_for_save}.cpt')


def show_and_save_figure(epochs, loss_train, loss_valid, save_name, minibatch_size):
    # 【iTerm2】ターミナル上で画像を表示する方法(https://qiita.com/noraworld/items/ea59c37e48ac0977cc72)
    sns.set()
    sns.set_style('whitegrid', {'gird.linestyle': '--'})
    sns.set_context("paper", 4, {"lines.linewidth": 6})
    sns.set_palette("pastel")

    x = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(1, 1, figsize=(30, 15), tight_layout=True)
    axes[0].plot(x, np.array(loss_train), label='loss_train')
    axes[0].plot(x, np.array(loss_valid), label='loss_valid')
    axes[0].set_title(f"loss(minibatch size={minibatch_size})")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    os.makedirs(os.path.join('work', 'figure', 'predicting_places'), exist_ok=True)
    save_path = os.path.join('work', 'figure', 'predicting_places', save_name+'.png')
    fig.savefig(save_path)
    plt.close()


def save_model(model):
    os.makedirs(os.path.join("/work02", "ariyama", "exp2022", "concept_neurons", "model", "predicting_places"),
                exist_ok=True)
    save_path_of_model_weights = os.path.join("/work02", "ariyama", "exp2022", "concept_neurons", "model", "predicting_places", "predicting_places_model_weights.pth")
    save_path_of_model = os.path.join("/work02", "ariyama", "exp2022", "concept_neurons", "model", "predicting_places", "predicting_places_model.pth")

    torch.save(model.state_dict(), save_path_of_model_weights)
    torch.save(model, save_path_of_model)


"""
# 正解率を計測する関数
def my_measure_accuracy(model, X, label):
    with torch.no_grad():
        logit = model(X)
        prob = nn.Softmax(dim=-1)(logit)  # 確率にする
        pred = torch.argmax(prob, dim=-1)  # 4つのラベルのうち最も高い確率を持つラベルを予測とする

        total = 0
        correct = 0
        for prediction, answer in zip(pred, label):
            if prediction == answer:
                correct += 1
            total += 1

        return correct * 100 / total
"""


def main():
    parser = argparse.ArgumentParser(description='save checkpoint each time a epoch ends')

    parser.add_argument('-lr', '--learning_rate', help='学習率(default=5e-4)', default=5e-4, type=float)
    parser.add_argument('-e', '--epochs', help='エポック数(default=100)', default=100, type=int)
    parser.add_argument('-s', '--seed', help='ランダムシード値(default=0)', default=0, type=int)
    parser.add_argument('-f', '--figure', help=f'図の保存名(default=predicting_places)', default=f'predicting_places', type=str)
    parser.add_argument('-m', '--mini_batch_size', help='ミニバッチサイズ(default=32)', default=32, type=int)
    parser.add_argument('-g', '--use_gpu', help='このフラグを指定すると，学習にGPUを必ず使用します', action='store_true')

    args = parser.parse_args()

    # logzero
    os.makedirs(os.path.join("log", "predicting_place"), exist_ok=True)
    log_file = os.path.join("log", "predicting_place", "predicting_place.log")
    logzero.logfile(log_file)
    logger.debug('start of script')
    print(f'log is saved in {log_file}')
    logger.debug('--------start of script--------')

    torch.manual_seed(args.seed)
    logger.info(f'random seed is {args.seed}')
    logger.info('')

    # GPUが使えるかどうかの判断
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # モデルの最適化プロセスを制御するためのハイパーパラメータ
    learning_rate = args.learning_rate
    epochs = args.epochs
    logger.info(f"learning_rate: {learning_rate}, {epochs}epochs")
    logger.info(f'mini_batch size: {args.mini_batch_size}')

    # モデル・損失関数・最適化器
    model = model_of_predicting_place.Model(in_features=768, intermediate_features=100, out_features=1)
    model.train()
    # モデルとクロスエントロピー関数をGPUに渡す
    model.to(device)
    if args.use_gpu:
        assert device == torch.device('cuda'), "couldn't use GPU"
    logger.info('Model:')
    logger.info(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(step_size=1)

    os.makedirs(os.path.join("work", "predicting_place"))
    file_path_of_loss_and_acc = os.path.join('work', 'predicting_place', 'predicting_place.txt')
    with open(file_path_of_loss_and_acc, 'w') as fi:
        fi.write('train_loss\tvalid_loss\n')
        train_loss_list = []
        valid_loss_list = []
        #train_acc_list = []
        #valid_acc_list = []
        total_time = 0.0
        min_loss_valid = 0.0
        for t in tqdm(range(epochs)):
            # データのロードをエポックごとにやることで、trainのシャッフルが毎回変わって欲しい
            train_dataloader, valid_dataloader, x_train, y_train, x_valid, y_valid = data_load(args.mini_batch_size)

            model.train()
            model.to(device)
            start = time.time()
            loss_train = train_loop(train_dataloader, model, criterion, optimizer, device)  # GPU上でのメモリ使用量はおよそ1000MiB
            total_time += (time.time() - start)

            # 1エポック終わるごとに推論(正解率の計測)を実行
            model.eval()
            with torch.no_grad():
                loss_valid = get_valid_loss(valid_dataloader, model, criterion)  # 評価モードに切り替えてから検証データのロス計算
                # 正解率を計測
                #acc_train = my_measure_accuracy(model, x_train, y_train)
                #acc_valid = my_measure_accuracy(model, x_valid, y_valid)

            train_loss_list.append(loss_train)
            valid_loss_list.append(loss_valid)
            #train_acc_list.append(acc_train)
            #valid_acc_list.append(acc_valid)
            fi.write(f'{loss_train}\t{loss_valid}\n')
            #fi.write(f'{loss_train}\t{loss_valid}\t{acc_train}\t{acc_valid}\n')

            if min_loss_valid > loss_valid:
                min_loss_valid = loss_valid

            if (t + 1) % 10 == 0:
                logger.info(f"epochs {t + 1}, loss {loss_train:>7f}")

            save_checkpoint(t + 1, model, optimizer, loss_train, args.epochs)

            with torch.no_grad():
                # グラフのプロットと保存
                show_and_save_figure(t + 1, train_loss_list, valid_loss_list, args.figure, args.mini_batch_size)

            # スケジューラを1ステップ進める
            # scheduler.step()

    logger.info(f"max accuracy of valid data: {min_loss_valid}%")
    logger.info(f"mini_batch size = {args.mini_batch_size} -> average: {total_time / epochs} sec / 1 epoch")
    logger.info("Done!")
    logger.info("")

    save_model(model)


if __name__ == "__main__":
    main()
