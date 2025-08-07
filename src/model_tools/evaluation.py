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
    
def eval_sero_pnn(dataset, model, N, n_samples=50):
    eval_loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    # Ensures dropout is active
    model.train()  
    with torch.no_grad():
        (obs, dow, sero_obs), y = next(iter(eval_loader))

        preds = np.zeros(shape=(dataset.__len__(), N,  n_samples))
        for i in range(n_samples):
            dist_pred, active_idxs, p_active = model(obs, dow, sero_obs)
            samples = np.zeros_like(active_idxs, dtype=np.float32)
            samples[active_idxs] = dist_pred.sample().numpy()
            preds[:, :, i] = samples
        return preds, y
    

    
def plot_pnn_preds(preds, dataset, title):
    preds_median = np.quantile(preds, 0.5, axis=1)
    preds_lower = np.quantile(preds, 0.025, axis=1)
    preds_upper = np.quantile(preds, 0.975, axis=1)

    y_true = [dataset.__getitem__(i)[1].item() for i in range(len(dataset))]
    dates = dataset.dates

    plt.plot(dates, preds_median, label=f'NowcastPNN Preds', color="red")
    plt.plot(dates, y_true, label=f'True y', color="black")
    plt.fill_between(dates, preds_lower, preds_upper, color='red', alpha=0.2, label='NowcastPNN 95% CI')

    
    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_prop_pnn_preds(preds, dataset, title):
    
    y_true = []
    for i in range(len(dataset)):
        _, z = dataset.__getitem__(i)
        y_true.append(z.sum())

    preds_median = np.quantile(preds, 0.5, axis=1)
    preds_lower = np.quantile(preds, 0.025, axis=1)
    preds_upper = np.quantile(preds, 0.975, axis=1)

    dates = dataset.dates

    plt.plot(dates, preds_median, label='PropPNN Preds', color="blue")
    plt.plot(dates, y_true, label=f'True y', color="black")
    plt.fill_between(dates, preds_lower, preds_upper, color='blue', alpha=0.2, label='PropPNN 95% CI')

    
    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_sero_pnn_preds(preds, dataset, N, title):
    preds_median = np.quantile(preds, 0.5, axis=-1)
    dates = dataset.dates
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    for s in range(N):
        y_true = [dataset.__getitem__(i)[1][s].item() for i in range(len(dataset))]

        plt.plot(dates, preds_median[:,s], label='SeroPNN Preds', color=colors[s])
        plt.plot(dates, y_true, label=f'True y', color="black")
        
        plt.legend()
        plt.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.title(f"{title}: DENV-{s+1}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


    
def eval_direct_sero(dataset, model, N, n_samples=50):
    eval_loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    # Ensures dropout is active
    model.train()  
    with torch.no_grad():
        (obs, dow, sero_obs), y = next(iter(eval_loader))

        preds = np.zeros(shape=(dataset.__len__(), N, n_samples))
        for i in range(n_samples):
            dist_pred = model(obs, dow, sero_obs)
            samples = dist_pred.sample().numpy()
            preds[:, :, i] = samples
        return preds, y