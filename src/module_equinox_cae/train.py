import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import trange
from typing import Union, Callable
import os

from io_utils import save_model, save_embeddings, save_train_losses, load_checkpoint, save_checkpoint



model_factory=Callable[[jax.random.PRNGKey], eqx.Module]

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
    """Function that trains an equinox autoencoder-like model with at least an encoder-like and decoder-like modules.

    Args:
        key (jnp.Random.PRNGKey): key for model; should be split from a primary key before passing.
        x_train (jnp.ndarray): Likely Noisy Inputs, but can be any adjustment to y_train as long as shape is the same.
        y_train (jnp.ndarray): Training outputs, likely clean images.
        x_test (jnp.ndarray): Idenctical to x_train but for validation.
        y_test (jnp.ndarray): Identical to y_train but for validation.
        epochs (int, optional): Number of total epochs for training (each epoch contains a certain amount of steps). Defaults to 10.
        batch_size (int, optional): Data Batch Sizes. Defaults to 128.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
        steps (int | None, optional): Number of steps in each epoch. If None its based on number of batches. Defaults to None.
        val_steps (int | None, optional): Number of validation steps in each Step. If None its based on number of validation batches. Defaults to None.
        autoencoder (Union[eqx.Module, model_factory, None], optional): The model tree either untrained or partiall trained. Defaults to None.
        loss_path (str | None, optional): Filepath for training and validation loss. Defaults to None.
        embeddings_path (str | None, optional): Filepath for post encoder embeddings. Defaults to None.
        final_model_path (str | None, optional): Filepath where final .eqx model is saved. Defaults to None.
        resume_from (str | None, optional): Filepath for tree containing information to restart partially trained model. Defaults to None.
        checkpoint_every_steps (int | None, optional): Frequency at which the periodic checkpoint saves. If None based on number of steps. Defaults to None.
        periodic_model_path (str | None, optional): Filepath at which the periodic checkpoints are saved. If None periodic checkpoints dont save. Defaults to None.
        best_model_path (str | None, optional): Filepath where the model with the best validation loss is saved. If None best validation model isn't saved. Defaults to None.

    Raises:
        ValueError: Checks for autoencoder argument.
        ValueError: Checks for final_model_path argument.
        ValueError: Checks for loss_path argument.
        ValueError: Checks for Embeddings_path argument.
        ValueError: Checks that the model has modules.
        ValueError: Checks that model has at least an encoder and decoder modules. (NEEDS ADJUSTING)
        ValueError: Checks that the partially trained model path exists

    Returns:
        eqx.Module: Final Model
        jnp.ndarray: Training Embeddings
        list[float]: Training Losses
        list[float]: Validation Losses
    """
    if steps is None: steps=max(1,len(x_train)//batch_size)
    if val_steps is None: val_steps=max(1,len(x_test)//batch_size)
    if checkpoint_every_steps is None: checkpoint_every_steps=max(1,steps//5)
    if autoencoder is None: raise ValueError("Please provide an autoencoder model as argument.")
    if final_model_path is None:raise ValueError("Please provide a path to save the final model.")
    if loss_path is None: raise ValueError("Please provide a path to save the training loss history.")
    if embeddings_path is None: raise ValueError("Please provide a path to save the training embeddings.")
    
    # model=autoencoder(key) if callable(autoencoder) else autoencoder
    model=autoencoder(key=key)

    # if not hasattr(model,"modules"): raise ValueError("The Provided model does not have 'modules' attribute.")
    # if hasattr(model,"modules") and len(model.modules)<=1: raise ValueError(
    #     f"The Provided model has only {len(model.modules)} module(s)."
    #     "Please provide a model with at least a separate encoder and decoder modules."
    #     "The first module must be considered the encoder and should produce plottable image embeddings."
    #     "The final module should be the decoder and should produce images."
    #     "The intermediate modules (if any) can be anything, but the results will not be stored.")

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y=jax.vmap(model)(x)
        return jnp.mean(jnp.clip(pred_y,0)-pred_y*y+jnp.log1p(jnp.exp(-jnp.abs(pred_y))))
    
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss,grads=compute_loss(model,x,y)
        updates,opt_state=optim.update(grads,opt_state,model)
        model=eqx.apply_updates(model,updates)
        return loss, model, opt_state

    def compute_metric(model,x,y):
        pred_y=jax.vmap(model)(x)
        losses=jnp.mean(jnp.clip(pred_y,0)-pred_y*y+jnp.log1p(jnp.exp(-jnp.abs(pred_y))))#-jnp.nanmean(y*jnp.log(pred_y)+(1-y)*jnp.log(1-pred_y))
        return losses

    optim=optax.adam(learning_rate)
    opt_state=optim.init(eqx.filter(model, eqx.is_array))

    start_epoch,global_step=0,0
    best_val=float('inf')

    if resume_from is not None:
        if os.path.exists(resume_from):
            model_like=model
            opt_like=opt_state
            key_like=key
            model,opt_state,start_epoch,global_step,key=load_checkpoint(resume_from,model_like,opt_like,key_like)
            print(f"Training Resumed from {resume_from} at the {start_epoch+1} epoch and {global_step+1} step.")
        else:
            raise ValueError(f"Checkpoint file {resume_from} not found.")
    
    train_losses: list[float]=[]
    val_losses: list[float]=[]
    print('start epochs')
    print(start_epoch)
    for epoch in range(start_epoch,epochs):
        bar=trange(steps)
        print('trange made?')
        for i in bar:
        # for i in range(steps):
            bar.set_description(f"Epoch {epoch+1}/{epochs}")
            start=i*batch_size
            end=start+batch_size

            loss,model,opt_state=make_step(model,x_train[start:end],y_train[start:end],opt_state)
            loss=loss.item()
            bar.set_postfix(train_loss=loss)
            train_losses.append(loss)
            global_step+=1
            

            if periodic_model_path is not None:
                if checkpoint_every_steps>0 and (global_step%checkpoint_every_steps ==0):
                    save_checkpoint(periodic_model_path, model, opt_state, epoch, global_step, key)
                    print(f"Checkpoint saved at step {global_step} to {periodic_model_path}")

            metrics=[]
            for j in range(val_steps):
                start=j*batch_size
                end=start+batch_size

                val_loss=compute_metric(model,x_test[start:end],y_test[start:end])
                metrics.append(val_loss)
            val_mean=jnp.nanmean(jnp.array(metrics))
            print(f"Epoch: {epoch+1} | Validation Loss: {val_mean:.3f}")
            if best_model_path is not None:
                if val_mean<best_val:
                    best_val=val_mean
                    save_checkpoint(best_model_path, model, opt_state, epoch, global_step, key)
                    print(f"New best model saved to {best_model_path}")
            val_losses.append(val_mean)
        print(f"Epoch {epoch+1}/{epochs} completed.")

    save_model(model, final_model_path)
    save_train_losses(train_losses,val_losses,loss_path)

    trained_encoder=model.modules[0]
    training_embeds=jax.vmap(trained_encoder)(x_train)
    training_embeds=jax.device_get(training_embeds)
    save_embeddings(jnp.asarray(training_embeds), embeddings_path)


    return model, training_embeds, train_losses, val_losses


    