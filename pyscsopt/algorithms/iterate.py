import numpy as np
from datetime import datetime

class Solution:
    def __init__(self, x, obj, fval, fvaltest, rel, objrel, metricvals, times, epochs, model, grad_norms=None):
        self.x = x
        self.obj = obj
        self.fval = fval
        self.fvaltest = fvaltest
        self.rel = rel
        self.objrel = objrel
        self.metricvals = metricvals
        self.times = times
        self.epochs = epochs
        self.model = model
        self.grad_norms = grad_norms

class Options:
    def __init__(self, metrics=None, alpha=None, batch_size=None, slice_samples=False, shuffle_batch=True, max_epoch=1, max_iter=None, comm_rounds=100, local_max_iter=None, x_tol=1e-10, f_tol=1e-10, verbose=1, patience=3):
        self.metrics = metrics
        self.alpha = alpha
        self.batch_size = batch_size
        self.slice_samples = slice_samples
        self.shuffle_batch = shuffle_batch
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.comm_rounds = comm_rounds
        self.local_max_iter = local_max_iter
        self.x_tol = x_tol
        self.f_tol = f_tol
        self.verbose = verbose
        self.patience = patience

def iter_step(method, model, reg_name, hmu, As, x, x_prev, ys, Cmat, iter):
    return method.step(model, reg_name, hmu, As, x, x_prev, ys, Cmat, iter)

def iterate(method, model, reg_name, hmu, metrics=None, alpha=None, batch_size=None, slice_samples=False, shuffle_batch=True, max_epoch=1000, comm_rounds=100, local_max_iter=None, x_tol=1e-10, f_tol=1e-10, verbose=1, patience=5):
    opt = Options(
        metrics=metrics,
        alpha=alpha,
        batch_size=batch_size,
        slice_samples=slice_samples,
        shuffle_batch=shuffle_batch,
        max_epoch=1 if local_max_iter is not None else max_epoch,
        comm_rounds=comm_rounds,
        local_max_iter=local_max_iter,
        x_tol=x_tol,
        f_tol=f_tol,
        verbose=verbose,
        patience=patience
    )
    return optim_loop(method, model, reg_name, hmu, opt)

def optim_loop(method, model, reg_name, hmu, opt):
    is_generic = (model.A is None or model.y is None)
    x = model.x0.copy()
    x_prev = x.copy()
    epochs = 0
    objs, fvals, rel_errors, objrels, times = [], [], [], [], []
    grad_norms = []
    t0 = datetime.now()
    # initialize the method with the starting point x (important for BFGS/Hessian-based methods)
    if method is not None and hasattr(method, 'init'):
        method.init(x)
    n = x.shape[0]
    patience_counter = 0
    best_fval = None
    for epoch in range(opt.max_epoch):
        if is_generic:
            fval = model.f(x)
            obj = fval + model.get_reg(x, reg_name)
        else:
            fval = model.f(model.A, model.y, x)
            obj = fval + model.get_reg(x, reg_name)
        fvals.append(fval)
        objs.append(obj)
        if hasattr(model, 'x_star'):
            rel_error = np.linalg.norm(x - model.x_star) / max(np.linalg.norm(model.x_star), 1)
        else:
            rel_error = np.linalg.norm(x - model.x0) / max(np.linalg.norm(model.x0), 1)
        rel_errors.append(rel_error)
        objrel = abs(obj - model.obj_star) / max(abs(model.obj_star), 1) if hasattr(model, 'obj_star') else 0.0
        objrels.append(objrel)
        times.append((datetime.now() - t0).total_seconds())
        if reg_name == "gl" and hasattr(model, "P") and hasattr(model.P, "Cmat"):
            Cmat = model.P.Cmat
        else:
            Cmat = np.eye(n)
        if best_fval is None or fval < best_fval:
            best_fval = fval
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= opt.patience:
            if opt.verbose > 0:
                print(f"Patience level {opt.patience} reached at epoch {epoch+1}, stopping early.")
            break
        if patience_counter < opt.patience - 1:
            if is_generic:
                x_new, grad_norm = iter_step(method, model, reg_name, hmu, None, x, x_prev, None, Cmat, epoch)
            else:
                x_new, grad_norm = iter_step(method, model, reg_name, hmu, model.A, x, x_prev, model.y, Cmat, epoch)
            grad_norms.append(grad_norm)
            if opt.verbose > 0:
                print(f"Epoch {epoch+1:4d} | obj: {obj:.6e} | fval: {fval:.6e} | rel_error: {rel_error:.2e} | objrel: {objrel:.2e} | grad_norm: {grad_norm:.2e}")
            if (
                np.linalg.norm(x_new - x) < opt.x_tol*max(np.linalg.norm(x), 1)
                or grad_norm < opt.x_tol * max(1, np.linalg.norm(x))
                ):
                break
            x_prev = x.copy()
            x = x_new.copy()
            epochs += 1
        else:
            # grad_norms.append(np.nan)
            break
    return Solution(x, objs, fvals, None, rel_errors, objrels, grad_norms, times, epochs, model, grad_norms=grad_norms)