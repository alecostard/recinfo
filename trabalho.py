import numpy as np
import pandas as pd
import sys
import uuid
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy


def now_str():
    return datetime.now().isoformat("-").replace(":", "-").replace(".", "-")


def load_ml_25m():
    return pd.read_csv("./datasets/ml-25m/ratings.csv", usecols=[0, 1, 2])


def load_ml_paper():
    return pd.read_csv("./datasets/paper/ratings.csv")


def resample_25m(minratings, nusers):
    data = load_ml_25m()
    x = data.groupby("userId").filter(lambda df: len(df) >= minratings)
    users = np.random.choice(x.userId.unique(), nusers, replace=False)
    return x[x.userId.isin(users)]


def load_simple_dataset():
    return pd.read_csv("./simple_dataset.csv")


def load_normal_dataset():
    return pd.read_csv("./dataset-200x1000.csv")


def dataframe_remove(original, toremove):
    return pd.concat([original, toremove]).drop_duplicates(keep=False)


def test_train_split(data, ratio):
    testdata = (data
                .groupby("userId", group_keys=False)
                .apply(lambda df: df.sample(int(ratio * len(df)))))

    traindata = dataframe_remove(data, testdata)

    return testdata, traindata


def make_testset(df):
    return list(df.itertuples(index=False, name=None))


def make_trainset(df):
    reader = Reader(rating_scale=(0.5, 5))
    return Dataset.load_from_df(df, reader).build_full_trainset()


def sample_traindata(data, n):
    return (data
            .groupby("userId", group_keys=False)
            .apply(lambda df: df.sample(min(len(df), n))))


def predictions_df(predictions):
    return pd.DataFrame(
        data=[[p.uid, p.iid, p.r_ui, p.est] for p in predictions],
        columns=["userId", "movieId", "rating", "estimate"])


def experiment(data, samples):
    np.random.seed()
    test, train = test_train_split(data, 0.5)
    testset = make_testset(test)

    predictions = single_run(99999, train, testset)

    for nsamples in samples:
        traindata = sample_traindata(train, nsamples)
        pred = single_run(nsamples, traindata, testset)
        predictions = predictions.append(pred)

    predictions["experimentId"] = uuid.uuid4()
    predictions.to_csv(f"./predictions/{now_str()}.csv", index=False)
    return predictions


def single_run(nsamples, traindata, testset):
    trainset = make_trainset(traindata)
    pred = predictions_df(SVD(n_factors=30).fit(trainset).test(testset))
    pred["samples"] = nsamples
    return pred


def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))


def mae(x, y):
    return np.mean(abs(x - y))


def normalize_rating(ratings):
    return (ratings - 0.5) / 4.5


def ndcg(df, at=10):
    amort = amort = np.array([np.log2(2 + i) for i in range(at)])

    ratings = df.sort_values("rating", ascending=False).head(at).rating.values
    nratings = normalize_rating(ratings)
    idcg = np.sum((2**nratings - 1) / amort)

    est = df.sort_values("estimate", ascending=False).head(at).rating.values
    nestimates = normalize_rating(est)
    dcg = np.sum((2**nestimates - 1) / amort)

    return dcg/idcg


def andcg(df, at=10):
    return df.groupby("userId").apply(lambda df: ndcg(df, at)).mean()


def global_metrics(predictions):
    results = (predictions
               .groupby('samples')
               .apply(lambda df: pd.Series([
                   rmse(df.rating, df.estimate),
                   andcg(df)]
               ))
               .reset_index())

    results.columns = ['samples', 'rmse', 'ndcg']
    expId = predictions['experimentId'].unique()[0]
    results['experimentId'] = expId
    results.to_csv(f"./metrics/global/{expId}.csv", index=False)
    return results


def user_metrics(predictions):
    results = (predictions
               .groupby(['samples', 'userId'])
               .apply(lambda df: pd.Series(
                   [rmse(df.rating, df.estimate), ndcg(df)]
               ))
               .reset_index())

    results.columns = ['samples', 'userId', 'rmse', 'ndcg']
    expId = predictions['experimentId'].unique()[0]
    results['experimentId'] = expId
    results.to_csv(f"./metrics/user/{expId}.csv", index=False)
    return results


def load_pred_and_user_metrics(filename):
    predictions = pd.read_csv(f"predictions/{filename}")
    user_metrics(predictions)


def load_pred_and_global_metrics(filename):
    predictions = pd.read_csv(f"predictions/{filename}")
    global_metrics(predictions)


def merge_all_files(dirname):
    return pd.concat(
        [pd.read_csv(f"{dirname}/{p}") for p in os.listdir(dirname)],
        ignore_index=True)


def run_experiments(nexp):
    samples = [1, 2, 4, 8, 16, 32, 50, 75, 100]
    data = load_normal_dataset()
    with Pool() as pool:
        pool.starmap(experiment, [(data, samples)]*nexp)


def run_user_metrics():
    preds = os.listdir("predictions")
    with Pool() as pool:
        pool.map(load_pred_and_user_metrics, preds)


def run_global_metrics():
    preds = os.listdir("predictions")
    with Pool() as pool:
        pool.map(load_pred_and_global_metrics, preds)


def plot_global_metrics():
    results = merge_all_files("metrics/global")
    sample_mean = results.groupby('samples').mean().reset_index()
    samples = sample_mean.samples[:-1]

    rmse = sample_mean.rmse[:-1]
    plt.figure()
    plt.plot(samples, rmse, '-x', label="RMSE por amostra")
    plt.axhline(sample_mean.rmse.values[-1], linestyle='--', label="Caso base")
    plt.xlabel("Amostras")
    plt.ylabel("RMSE")
    plt.legend()

    ndcg = sample_mean.ndcg[:-1]
    plt.figure()
    plt.plot(samples, ndcg, '-x', label="nDCG por amostra")
    plt.axhline(sample_mean.ndcg.values[-1], linestyle='--', label="Caso base")
    plt.ylim(top=1.02*sample_mean.ndcg.values[-1])
    plt.xlabel("Amostras")
    plt.ylabel("nDCG")
    plt.legend()

    plt.show()


def plot_one_sorted_global(label):
    results = merge_all_files("metrics/user")
    groups = results.groupby(['samples', 'userId']).mean().reset_index()
    samples = np.sort(results.samples.unique())[:-1]

    plt.figure()
    base = groups[groups.samples == 99999].sort_values(label)[label].values
    plt.plot(base, 'k', label="Caso base")

    for sample in samples:
        base = groups[groups.samples == sample].sort_values(label)[
            label].values
        plt.plot(base, label=f"n = {sample}", linewidth=0.5)

    plt.xlabel("Usuário")
    plt.ylabel(f"{label.upper()}")
    plt.legend()
    plt.show()


def plot_one_user_metric(label):
    results = merge_all_files("metrics/user")
    groups = results.groupby(['samples', 'userId']).mean().reset_index()
    samples = np.sort(results.samples.unique())[:-1]
    data = (groups[groups.samples == 99999][["userId", label]]
            .rename(columns={label: "full"}))

    for x in samples:
        current = (groups[groups.samples == x][["userId", label]]
                   .rename(columns={label: x}))
        data = data.merge(current, on="userId")

    data = data.sort_values('full')

    plt.figure()
    plt.plot(data['full'].values, 'k', label="Caso base")
    plt.plot(data[1].values, label="1 amostra", linewidth=0.5)
    plt.plot(data[100].values, label="100 amostras", linewidth=0.5)
    plt.xlabel("Usuário")
    plt.ylabel(f"{label.upper()}")
    plt.legend()


def plot_user_metrics():
    for metric in ['ndcg', 'rmse']:
        plot_one_user_metric(metric)
    plt.show()
