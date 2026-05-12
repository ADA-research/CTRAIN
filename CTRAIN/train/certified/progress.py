from tqdm.auto import tqdm


def _as_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def format_eps(value):
    value = _as_float(value)
    return f"{value:.4f}"


def progress_bar(iterable, epoch, num_epochs, eps, method=None, disable=False):
    prefix = f"Epoch {epoch + 1}/{num_epochs}"
    if method:
        prefix = f"{method} {prefix}"
    return tqdm(
        iterable,
        total=len(iterable),
        desc=f"{prefix} eps={format_eps(eps)}",
        leave=False,
        dynamic_ncols=True,
        disable=disable,
    )


def update_progress(progress, loss, nat_acc=None, cert_acc=None, adv_acc=None, lr=None):
    values = {"loss": f"{_as_float(loss):.4f}"}
    if nat_acc is not None:
        values["nat"] = f"{_as_float(nat_acc):.3f}"
    if adv_acc is not None:
        values["adv"] = f"{_as_float(adv_acc):.3f}"
    if cert_acc is not None:
        values["cert"] = f"{_as_float(cert_acc):.3f}"
    if lr is not None:
        values["lr"] = f"{_as_float(lr):.2e}"
    progress.set_postfix(values)


def log_epoch_summary(epoch, num_epochs, loss, nat_acc, cert_acc, adv_acc=None):
    parts = [
        f"Epoch [{epoch + 1}/{num_epochs}]",
        f"loss={_as_float(loss):.4f}",
        f"nat_acc={_as_float(nat_acc):.4f}",
    ]
    if adv_acc is not None:
        parts.append(f"adv_acc={_as_float(adv_acc):.4f}")
    parts.append(f"cert_acc={_as_float(cert_acc):.4f}")
    tqdm.write(", ".join(parts))
