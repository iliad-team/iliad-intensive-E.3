# %%


import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time
import random, numpy
from dataclasses import dataclass
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

# %%

@dataclass
class Parameters:
    n_ctx: int = 2
    d_vocab: int = 2048
    d_model: int = 128
    num_epoch: int = 2
    batch_size: int = 1024
    subset_percentage: float = 5  # each epoch draws this % of all d_vocab**n_ctx possible inputs
    lr: float = 0.001


params = Parameters()
performance = {}  # proof name -> (loss bound, seconds)

# Run everything on the GPU if there is one (Runtime > Change runtime type > GPU on Colab)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%

def set_seed(seed: int = 57) -> None:
    numpy.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

# %%

def measure_time(func):
    def wrapper(*args, **kwargs):
        if device.type == "cuda":
            t.cuda.synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        if device.type == "cuda":
            t.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in: {elapsed_time:.6f} seconds")
        return result, elapsed_time

    return wrapper

# %%

set_seed(57)

# %%

def training_step(
    model,
    optimizer,
    inputs: Int[Tensor, "batch_size n_ctx"],
    labels: Int[Tensor, "batch_size"],
    params: Parameters,
):

    criterion = t.nn.CrossEntropyLoss()

    inputs_one_hot = F.one_hot(inputs, params.d_vocab).float()

    outputs = model(inputs_one_hot)

    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss

# %%

def train(model, params):

    loss_history = []

    n_samples = int(params.d_vocab**params.n_ctx * (params.subset_percentage / 100))

    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=params.lr,
    )

    set_seed(57)
    for epoch in tqdm(range(params.num_epoch)):
        # Fresh random inputs every epoch; the label is the max of each row
        inputs = t.randint(0, params.d_vocab, (n_samples, params.n_ctx), device=device)
        labels = inputs.max(dim=1).values

        for batch_inputs, batch_labels in zip(
            inputs.split(params.batch_size), labels.split(params.batch_size)
        ):
            loss = training_step(
                model=model,
                optimizer=optimizer,
                inputs=batch_inputs,
                labels=batch_labels,
                params=params,
            )

            loss_history.append(loss.detach().item())

    return loss_history

# %%

class MLP(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_ctx = params.n_ctx

        self.embedding = t.nn.Linear(params.d_vocab, params.d_model, bias=False)
        self.linear = t.nn.Linear(params.d_model, params.d_model, bias=False)
        self.unembedding = t.nn.Linear(params.d_model, params.d_vocab, bias=False)

    def g(self, x):

        return self.unembedding((self.linear(x)))

    def forward(self, a):

        return self.g(self.embedding(a.sum(dim=1)))

# %%

model = MLP(params=params)

# %%

model = MLP(params=params).to(device)
loss_history = train(model=model, params=params)

# %%

plt.plot(loss_history)
plt.title("Loss Curve")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# %%

loss_history[-5:]

# %%

@measure_time
def brute_force_loss_proof(model, params):
    loss = 0
    criterion = t.nn.CrossEntropyLoss()

    with t.no_grad():
        model.eval()

        for x in tqdm(range(0, params.d_vocab)):

            x_tensor = t.full((params.d_vocab,), x, device=device)
            y_tensor = t.arange(params.d_vocab, device=device)

            labels = t.max(x_tensor, y_tensor)

            inputs = t.stack(
                [
                    F.one_hot(x_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(y_tensor, num_classes=params.d_vocab).float(),
                ],
                dim=1,
            )

            outputs = model(inputs)

            loss += criterion(outputs, labels)

    return loss / params.d_vocab

# %%

loss_bf, time_bf = brute_force_loss_proof(model=model, params=params)

# %%

performance["Brute force"] = (loss_bf, time_bf)

# %%

@measure_time
def symmetry_proof_loss(model, params):
    loss = 0
    criterion = t.nn.CrossEntropyLoss()

    with t.no_grad():
        model.eval()
        for x in tqdm(range(0, params.d_vocab - 1)):

            x_tensor = t.full((params.d_vocab - x - 1,), x, device=device)
            y_tensor = t.arange(x + 1, params.d_vocab, device=device)

            labels = t.max(x_tensor, y_tensor)
            inputs = t.stack(
                [
                    F.one_hot(x_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(y_tensor, num_classes=params.d_vocab).float(),
                ],
                dim=1,
            )

            outputs = model(inputs)

            loss += criterion(outputs, labels) * 2 * len(x_tensor)

        x_tensor = t.eye(params.d_vocab, device=device)
        inputs = t.stack([x_tensor] * 2, dim=1)

        outputs = model(inputs)

        loss += criterion(outputs, t.arange(params.d_vocab, device=device)) * len(
            x_tensor
        )

    return loss / (params.d_vocab * params.d_vocab)

# %%

loss_sym, time_sym = symmetry_proof_loss(model=model, params=params)

# %%

performance["Symmetric"] = (loss_sym, time_sym)

# %%

@measure_time
def convexity_proof(model, params):
    loss = 0

    with t.no_grad():
        model.eval()

        criterion = t.nn.CrossEntropyLoss()

        inputs = t.stack([t.eye(params.d_vocab, device=device) * 2], dim=1)

        logits = model(inputs)

        for i in tqdm(range(1, params.d_vocab)):

            # sum_{t1 < i} L(g(2 t1.E), i)  +  i copies of L(g(2 i.E), i)
            loss += i * criterion(logits[:i], t.full((i,), i, device=device))
            loss += i * criterion(logits[i : i + 1], t.full((1,), i, device=device))

        loss += params.d_vocab * criterion(
            logits, t.arange(params.d_vocab, device=device)
        )

    return loss / (params.d_vocab**2)

# %%

performance["Convex"] = convexity_proof(model=model, params=params)

# %%

def plot_performance(performance, title):
    colors = {"Brute force": "red", "Symmetric": "green", "Convex": "blue"}

    for label, (loss, elapsed_time) in performance.items():
        if loss is None:  # exercise not implemented yet
            continue
        plt.scatter(elapsed_time, float(loss), color=colors[label], label=label)

    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.title(title)
    plt.xlabel("Time needed (in seconds)")
    plt.ylabel("Loss estimate")
    plt.show()


plot_performance(performance, "Different proof strategies to upper bound loss")

# %%

params_3 = Parameters(n_ctx=3, d_vocab=256)
performance_3 = {}
set_seed(57)
model_3 = MLP(params=params_3).to(device)

# %%

loss_history_3 = train(model=model_3, params=params_3)

# %%

plt.plot(loss_history_3)
plt.title("Loss Curve")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# %%

@measure_time
def brute_force_loss_proof_3(model, params):
    loss = 0
    criterion = t.nn.CrossEntropyLoss(reduction="sum")

    count = 0

    with t.no_grad():
        model.eval()

        # the first two tokens run over all d_vocab^2 pairs; these don't depend on x
        x_tensor = t.arange(params.d_vocab, device=device).repeat_interleave(params.d_vocab)
        y_tensor = t.arange(params.d_vocab, device=device).repeat(params.d_vocab)
        x_one_hot = F.one_hot(x_tensor, num_classes=params.d_vocab).float()
        y_one_hot = F.one_hot(y_tensor, num_classes=params.d_vocab).float()

        for x in tqdm(range(0, params.d_vocab)):

            z_tensor = t.full((params.d_vocab**2,), x, device=device)

            max_xy = t.max(x_tensor, y_tensor)
            labels = t.max(max_xy, z_tensor)

            inputs = t.stack(
                [
                    x_one_hot,
                    y_one_hot,
                    F.one_hot(z_tensor, num_classes=params.d_vocab).float(),
                ],
                dim=1,
            )

            outputs = model(inputs)

            loss += criterion(outputs, labels)

    return loss / params.d_vocab**3

# %%

performance_3["Brute force"] = brute_force_loss_proof_3(model=model_3, params=params_3)
performance_3["Brute force"][0]

# %%

def convexity_proof_three_equal(
    model,
    params: Parameters,
):

    loss = 0
    criterion = t.nn.CrossEntropyLoss(reduction="sum")

    with t.no_grad():

        # Estimate f(x,x,x)

        x_one_hot = t.eye(params.d_vocab, device=device)

        inputs = t.stack([x_one_hot] * 3, dim=1)
        outputs = model(inputs)

        loss += criterion(outputs, t.arange(params.d_vocab, device=device))

    return loss

# %%

def convexity_proof_two_equal(model, params: Parameters):

    loss = 0
    criterion = t.nn.CrossEntropyLoss(reduction="sum")

    with t.no_grad():
        # Estimate f(x,x,z) where x<z
        for z in range(1, params.d_vocab):

            # [0,1,...,z-1]
            x_tensor = t.arange(z, device=device)
            # [z,z,...,z]
            z_tensor = t.full((z,), z, device=device)

            inputs = t.stack(
                [
                    F.one_hot(x_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(x_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(z_tensor, num_classes=params.d_vocab).float(),
                ],
                dim=1,
            )
            outputs = model(inputs)

            loss += 3 * criterion(outputs, z_tensor)

        # Estimate f(x,z,z) where x<z
        for z in range(1, params.d_vocab):

            # [0,1,...,z-1]
            x_tensor = t.arange(z, device=device)
            # [z,z,...,z]
            z_tensor = t.full((z,), z, device=device)

            inputs = t.stack(
                [
                    F.one_hot(x_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(z_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(z_tensor, num_classes=params.d_vocab).float(),
                ],
                dim=1,
            )
            outputs = model(inputs)

            loss += 3 * criterion(outputs, z_tensor)

    return loss

# %%

@measure_time
def convexity_proof_3(model, params: Parameters):

    loss = []
    criterion = t.nn.CrossEntropyLoss(reduction="sum")

    with t.no_grad():
        # Estimate f(x,y,z) where x<y<z

        def estimate_fixed(z: int):
            count = 0

            # [0,1,...,z-1]
            x_tensor = t.arange(z, device=device)

            length = x_tensor.size(dim=0)

            # [z,z,...,z]
            z_tensor = t.full((length,), z, device=device)

            inputs = t.stack(
                [
                    F.one_hot(x_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(x_tensor, num_classes=params.d_vocab).float(),
                    F.one_hot(z_tensor, num_classes=params.d_vocab).float(),
                ],
                dim=1,
            )

            outputs = model(inputs)

            count = (length - 1) * criterion(outputs, z_tensor)

            return 3 * count

        for z in tqdm(range(2, params.d_vocab)):
            loss.append(estimate_fixed(z))

    loss.append(convexity_proof_three_equal(model=model, params=params))

    loss.append(convexity_proof_two_equal(model=model, params=params))

    return sum(loss) / params.d_vocab**3

# %%

performance_3["Convex"] = convexity_proof_3(model=model_3, params=params_3)
performance_3["Convex"][0]

# %%

plot_performance(performance_3, "Different proof strategies to upper bound loss (max of 3)")

# %%
