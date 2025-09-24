import os
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from jonah.module_equinox_cae.src.module_equinox_cae.io_utils import load_embeddings, load_predictions, load_train_losses

# ============================================================================ #
# plot preds
# ============================================================================ #
def plot_imgs(key:jax.random.PRGKey,title:str,data_name:str, n:int,
               x_data:jnp.ndarray, y_data:jnp.ndarray,
               embeddings:jnp.ndarray, predictions:jnp.ndarray):
    idx=jax.random.randint(key,(n,),0,len(x_data))
    x_imgs=x_data[idx,:]
    y_imgs=y_data[idx,:]
    embed_imgs=embeddings[idx,:]
    pred_imgs=predictions[idx,:]

    fig=plt.figure(figsize=(20,12),layout="tight")
    plt.suptitle(f"Images from {data_name} - {title}", fontsize=16)

    for i, (x_img, y_img, embed_img, pred_img) in enumerate(zip(x_imgs,y_imgs,embed_imgs,pred_imgs)):
        ax=plt.subplot(4,n,i+1)
        plt.imshow(y_img.reshape(28,28))
        plt.plasma()
        ax.set_axis_off()
        if i==0:
            ax.set_title("Target Data",loc="left",fontsize=10)
        
        ax=plt.subplot(4,n,i+1+n)
        plt.imshow(x_img.reshape(28,28))
        plt.plasma()
        ax.set_axis_off()
        if i==0:
            ax.set_title("Input Data",loc="left",fontsize=10)

        ax=plt.subplot(4,n,i+1+2*n)
        plt.imshow(embed_img.reshape(28,28))
        plt.plasma()
        ax.set_axis_off()
        if i==0:
            ax.set_title("Embeddings",loc="left",fontsize=10)
        
        ax=plt.subplot(4,n,i+1+3*n)
        plt.imshow(pred_img.reshape(28,28))
        plt.plasma()
        ax.set_axis_off
        if i==0:
            ax.set_title("Predictions",loc="left",fontsize=10)

    return fig

def plot