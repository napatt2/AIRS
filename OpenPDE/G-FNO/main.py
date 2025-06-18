# train_gfno2d.py
import os
import random
import datetime
from timeit import default_timer

import numpy as np
import torch
import h5py
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.GFNO import GFNO2d
from utils import pde_data, LpLoss, eq_check_rt, eq_check_rf
from scipy.io import loadmat

# -----------------------------------------------------------------------------
# 1) CONFIGURATION (edit these paths & hyper‐parameters)
# -----------------------------------------------------------------------------
DATA_PATH      = r"C:\Users\napat\Box\Research\ML\FNO\Dataset\ns_v1e-3_T50_2.mat"#./data/ns_V1e-4_N10000_T30.mat"
RESULTS_DIR    = "./results/gfno2d_run"
SEED           = 1

# model / training hyper‐parameters
T_IN           = 50        # number of input timesteps
T_OUT          = 50        # number of output timesteps
SPATIAL_RES    = 64        # Sx = Sy
N_TRAIN        = 8
N_VALID        = 1
N_TEST         = 1
BATCH_SIZE     = 1
EPOCHS         = 100
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
EARLY_STOP     = 20        # stop if valid loss doesn't improve for this many epochs
MODES          = 12
WIDTH          = 10
GRID_TYPE      = "symmetric"  # or "cartesian", None
STRATEGY       = "oneshot" #"teacher_forcing","markov", "recurrent", "oneshot"

# -----------------------------------------------------------------------------
# 2) REPRODUCIBILITY
# -----------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# 3) CREATE OUTPUT & LOGGING
# -----------------------------------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
run_id    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir   = os.path.join(RESULTS_DIR, f"gfno2d_{run_id}")
os.makedirs(out_dir, exist_ok=True)
writer    = SummaryWriter(out_dir)
MODEL_SAVE= os.path.join(out_dir, "best_model.pt")
#%%
# -----------------------------------------------------------------------------
# 4) LOAD DATA
# -----------------------------------------------------------------------------
'''
# Assumes MATLAB .mat file with variable 'u' of shape [Nt, Sy, Sx]
with h5py.File(DATA_PATH, 'r') as f:
    u = np.array(f['u_new'])
# transpose if needed so that u.shape == (Nt, Sy, Sx)
u = np.transpose(u, axes=(0,2,1))  # adjust depending on your file

# add channel dimension
data = torch.from_numpy(u[..., None].astype(np.float32))

assert data.shape[0] >= N_TRAIN + N_VALID + N_TEST + T_IN + T_OUT
'''
#%%
f = loadmat(DATA_PATH)
u = np.array(f['u_new'])
# transpose if needed so that u.shape == (Nt, Sy, Sx)
u = np.transpose(u, axes=(0,2,1))  # adjust depending on your file

# add channel dimension
data = torch.from_numpy(u[..., None].astype(np.float32))

assert data.shape[0] >= N_TRAIN + N_VALID + N_TEST + T_IN + T_OUT
#%%
# split
train_data = data[:N_TRAIN + T_IN + T_OUT]
valid_data = data[N_TRAIN : N_TRAIN + N_VALID + T_IN + T_OUT]
test_data  = data[-(N_TEST + T_IN + T_OUT):]

# wrap into pde_data (creates (input, target) pairs)
train_ds = pde_data(train_data, train=True,  strategy=STRATEGY, T_in=T_IN, T_out=T_OUT)
valid_ds = pde_data(valid_data, train=False, strategy=STRATEGY, T_in=T_IN, T_out=T_OUT)
test_ds  = pde_data(test_data,  train=False, strategy=STRATEGY, T_in=T_IN, T_out=T_OUT)
#%%
x_train = u[:N_TRAIN,:,:] #u[:N_TRAIN,:,:,0:1]
y_train = u[:N_TRAIN,:,:]

x_valid = u[N_TRAIN:N_TRAIN+N_VALID,:,:] #u[N_TRAIN:N_TRAIN+N_VALID,:,:,0:1]
y_valid = u[N_TRAIN:N_TRAIN+N_VALID,:,:]

x_test = u[N_TRAIN+N_VALID:,:,:] #u[N_TRAIN+N_VALID:,:,:,0:1]
y_test = u[N_TRAIN+N_VALID:,:,:]

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

x_valid = torch.from_numpy(x_valid).float()
y_valid = torch.from_numpy(y_valid).float()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()
#%%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------------------------------------------------------
# 5) BUILD MODEL
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GFNO2d(num_channels, initial_step, modes, width, reflection=False, grid_type=None)
initial_step = T_IN if STRATEGY in ("recurrent","teacher_forcing") else 1
model = GFNO2d(
    num_channels=50,
    initial_step=initial_step,
    modes=MODES,
    width=WIDTH,
    reflection=False,
    grid_type=GRID_TYPE
).to(device)

# equivariance sanity check on random input
x0 = torch.randn(BATCH_SIZE, SPATIAL_RES, SPATIAL_RES, T_IN, 1).to(device)
print("Rot equiv.:", eq_check_rt(model, x0, spatial_dims=[1,2]))
print("Refl equiv.:", eq_check_rf(model, x0, spatial_dims=[1,2]))

# -----------------------------------------------------------------------------
# 6) OPTIMIZER & LOSS
# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(train_loader)
)
criterion = LpLoss(size_average=False)
#%%
# -----------------------------------------------------------------------------
# 7) TRAINING LOOP
# -----------------------------------------------------------------------------
best_val = float('inf')
no_improve = 0

for ep in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred.reshape(len(pred), -1,50),
                         y_batch.reshape(len(y_batch),-1,50))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xv, yv in valid_loader:
            xv, yv = xv.to(device), yv.to(device)
            pv = model(xv)
            val_loss += criterion(pv.reshape(len(pred), -1,50),
                                  yv.reshape(len(y_batch),-1,50)).item()
    val_loss /= len(valid_loader)

    writer.add_scalar("Loss/Train", train_loss, ep)
    writer.add_scalar("Loss/Valid", val_loss,  ep)
    print(f"Epoch {ep:03d} — Train: {train_loss:.4e} | Valid: {val_loss:.4e}")

    # early stopping
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_SAVE)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= EARLY_STOP:
            print(f"No improvement for {EARLY_STOP} epochs — stopping.")
            break

# -----------------------------------------------------------------------------
# 8) TEST
# -----------------------------------------------------------------------------
model.load_state_dict(torch.load(MODEL_SAVE))
model.eval()
test_loss = 0.0
with torch.no_grad():
    for xt, yt in test_loader:
        xt, yt = xt.to(device), yt.to(device)
        pt = model(xt)
        test_loss += criterion(pt.view(len(pt),-1,1),
                               yt.view(len(yt),-1,1)).item()
test_loss /= len(test_loader)
print(f"TEST LOSS: {test_loss:.4e}")

writer.add_scalar("Loss/Test", test_loss)
writer.close()
