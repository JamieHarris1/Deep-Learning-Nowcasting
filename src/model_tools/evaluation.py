import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def eval_pnn(dataset, model, n_samples=50):
    eval_loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    # Ensures dropout is active
    model.train()  
    with torch.no_grad():
        (obs, dow), y = next(iter(eval_loader))

        preds = np.zeros(shape=(dataset.__len__(), n_samples))
        for i in range(n_samples):
            dist_pred = model(obs, dow)
            samples = dist_pred.sample().numpy()
            preds[:, i] = samples
        return preds, y

def eval_prop_pnn(dataset, model, n_samples=50):
    eval_loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    # Ensures dropout is active
    model.train()  
    with torch.no_grad():
        (obs, dow), z = next(iter(eval_loader))
        preds = np.zeros(shape=(dataset.__len__(), n_samples))
        for i in range(n_samples):
            dist_pred = model(obs, dow)
            samples = dist_pred.sample().numpy()
            preds[:, i] = samples.sum(1)
        return preds, z
    
def eval_sparse_prop_pnn(dataset, model, n_samples=50):
    eval_loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    # Ensures dropout is active
    model.train()  
    with torch.no_grad():
        (obs, dow), z = next(iter(eval_loader))

        preds = np.zeros(shape=(dataset.__len__(), n_samples))
        for i in range(n_samples):
            dist_pred, active_idxs = model(obs, dow)
            samples = np.zeros_like(active_idxs, dtype=np.float32)
            samples[active_idxs] = dist_pred.sample().numpy()
            preds[:, i] = samples.sum(-1)
        return preds, z
    
def plot_pnn_preds(preds, dataset, title):
    preds_median = np.quantile(preds, 0.5, axis=1)
    y_true = [dataset.__getitem__(i)[1].item() for i in range(len(dataset))]
    dates = dataset.dates

    plt.plot(dates, preds_median, label=f'PNN Preds')
    plt.plot(dates, y_true, label=f'True y', color="black")
    
    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_prop_pnn_preds(preds, dataset, title):
    preds_median = np.quantile(preds, 0.5, axis=1)
    y_true = []
    for i in range(len(dataset)):
        _, z = dataset.__getitem__(i)
        y_true.append(z.sum())

    dates = dataset.dates

    plt.plot(dates, preds_median, label='PropPNN Preds')
    plt.plot(dates, y_true, label=f'True y', color="black")
    
    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


    
    