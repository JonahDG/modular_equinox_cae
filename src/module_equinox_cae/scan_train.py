import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import scan_tqdm
import equinox as eqx
import optax
from typing import Union,Callable
import os

from io_utils import (
    save_model,
    save_embeddings,
    save_train_losses,
    load_checkpoint,
    save_checkpoint
)

model_factory=Callable[[jax.random.PRNGKey], eqx.Module]



def bce_logits(logits, y):
    return jnp.mean(jnp.clip(logits,0)-logits*y+jnp.log1p(jnp.exp(-jnp.abs(logits))))

def batch_data(x,y,batch_size):
    n=(x.shape[0]//batch_size)*batch_size
    x=x[:n]
    y=y[:n]

    steps=n//batch_size
    new_shape_x=(steps,batch_size)+x.shape[1:]
    new_shape_y=(steps,batch_size)+y.shape[1:]
    
    return x.reshape(new_shape_x), y.reshape(new_shape_y),steps

def train(
        key,
        x_train, y_train, x_test, y_test,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        steps: int | None=None,
        val_steps: int | None=None,
        autoencoder: Union[eqx.Module, model_factory, None]=None,
        *,
        loss_path: str | None=None,
        embeddings_path: str | None=None,
        final_model_path: str | None=None,
        resume_from: str | None=None,
        checkpoint_every_steps: int | None=None,
        periodic_model_path: str | None=None,
        best_model_path: str| None=None
        ):
    model=autoencoder(key=key)
    # print('test')
    params,static=eqx.partition(model,eqx.is_array)
    optim=optax.adam(learning_rate)
    opt_state=optim.init(params)

    xb_train,yb_train,train_steps=batch_data(x_train,y_train,batch_size)
    xb_test,yb_test,test_steps=batch_data(x_test,y_test,batch_size)

    @eqx.filter_value_and_grad
    def loss_fn(params,x,y):
        model=eqx.combine(params,static)
        logits=jax.vmap(model)(x)
        return bce_logits(logits,y)

    @eqx.filter_jit(donate="all")
    def train_epoch(params,opt_state,xb,yb,epoch_num,epochs=epochs):
        def step(carry,batch):
            params,opt_state=carry
            x,y=batch
            loss,grads=loss_fn(params,x,y)
            updates,opt_state=optim.update(grads,opt_state,params)
            params=eqx.apply_updates(params,updates)
            return (model,opt_state), loss
        # scan_tqdm_bar=scan_tqdm(xb.shape[0],desc=f"Epoch {epoch_num} of {epochs}")

        # (model,opt_state),losses=lax.scan(scan_tqdm_bar(step),(params,opt_state),(xb,yb))
        (params,opt_state),losses=lax.scan(step,(params,opt_state),(xb,yb))
        return params,opt_state,losses
    
    @eqx.filter_jit
    def val_epoch(params,xb,yb):
        model=eqx.combine(params,static)
        def step(sum_loss,batch):
            x,y=batch
            logits=jax.vmap(model)(x)
            loss=bce_logits(logits,x)
            return acc_loss+loss,None
        total_loss,_=lax.scan(step,0.0,(xb,yb))
        return total_loss/xb.shape[0]
    
    start_epoch=0
    best_val=jnp.inf
    if resume_from is not None:
        if os.path.exists(resume_from):
            model_like=model
            opt_like=opt_state
            key_like=key
            model,opt_state,start_epoch,load_checkpoint(resume_from,model_like,opt_like,key_like)
            print(f"Training Resumed from {resume_from} at the {start_epoch+1} epoch and {global_step+1} step.")
        else:
            raise ValueError(f"Checkpoint file {resume_from} not found")
    
    epoch_train_losses=[]
    epoch_val_losses=[]

    for epoch in range(start_epoch,epochs):
        params,opt_state, train_losses=train_epoch(params, opt_state, xb_train, yb_train,epoch)
        val_loss = val_epoch(params, xb_test, yb_test)

        train_losses=jax.device_get(train_losses)
        val_loss=jax.device_get(val_loss)

        train_loss_matrix.append(train_losses)
        val_loss_matrix.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_losses.mean():.6f} | val_loss={val_loss:.6f}")

        if best_model_path is not None:
            if val_loss < best_val:
                best_val=val_loss
                best_model=eqx.combine(params,static)
                save_model(best_model,best_model_path)
    final_model=eqx.combine(params,static)
    save_model(final_model, final_model_path)
    save_train_losses(train_losses,val_losses,loss_path)

    trained_encoder=model.modules[0]
    training_embeds=jax.vmap(trained_encoder)(x_train)
    training_embeds=jax.device_get(training_embeds)
    save_embeddings(jnp.asarray(training_embeds), embeddings_path)

    return model, training_embeds, train_losses, val_losses

