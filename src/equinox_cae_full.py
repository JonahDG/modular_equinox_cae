# ============================================================================ #
#  imports
# ============================================================================ #
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
import matplotlib.pyplot as plt
import optax
from tqdm import trange
from time import time
import argparse

# ============================================================================ #
# Arugments
# ============================================================================ #

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data", help="data name")
    p.add_argument("--model", default=None, help="Name of full autoencoder model assuming its in the path /fred/oz440/jonah/equinox_cae/models/DATA_MODEL.eqx and contains both a encoder and decoder modules")
    # p.add_argument("--time", default=False, help="If True, will time the encode and decode. False by default") # DEPRECATED
    p.add_argument("--epochs", type=int,default=10, help="Number of epochs")
    args=p.parse_args()
    return args

# ============================================================================ #
# MULTI CLASS MODEL
# ============================================================================ #

# ============================================================================ #
# Encode
class autoencoder_encode(eqx.Module):
    layers: list

    def __init__(self, *, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, key=key1),
            jax.nn.relu,
            # jax.nn.leaky_relu,
            eqx.nn.MaxPool2d((2, 2), 2),
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, key=key2),
            jax.nn.relu,
            # jax.nn.leaky_relu,
            eqx.nn.MaxPool2d((2, 2), 2),
            eqx.nn.Lambda(lambda z: z.reshape(*z.shape[:-3], -1)),
            # eqx.nn.Lambda(lambda z:jnp.moveaxis(z,0,-1)),
            eqx.nn.Linear(1568, 128, key=key3),
            # jax.nn.leaky_relu,
            jax.nn.relu,
            eqx.nn.Linear(128,1568,key=key4),
            # jax.nn.leaky_relu,
            jax.nn.relu,
            eqx.nn.Lambda(lambda z: z.reshape(*z.shape[:-1],32,7,7))
            # eqx.nn.Lambda(lambda z:jnp.moveaxis(z,-1,0))
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        return x

# ============================================================================ #
# Decode
class autoencoder_decode(eqx.Module):
    layers: list
    
    def __init__(self, *, key):
        _, _, _, _, key5, key6, key7 = jax.random.split(key,7)
        self.layers = [
            eqx.nn.ConvTranspose(2, in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, key=key5),
            jax.nn.relu,
            eqx.nn.ConvTranspose(2, in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, key=key6),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, key=key7),
            # jax.nn.sigmoid
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================ #
# Combined
class autoencoder(eqx.Module):
    modules: list
    def __init__(self, *, key):
        encoder = autoencoder_encode(key=key)
        decoder = autoencoder_decode(key=key)
        self.modules = [encoder, decoder]
    def __call__(self, x):
        for layer in self.modules:
            x = layer(x)
        return x

# ============================================================================ #
# Training
def train(x_train,y_train,x_test,y_test,epochs,key,batch_size=128,learning_rate=0.001,steps=60000//128,val_steps=10000//128):
    """Trains combined autoencoder
    Args:
        x_train: x training data (Likely Noisy)
        y_tain: y training data
        x_test: x test data (Likely Noisy)
        y_test: y test data
        epoch: Number of epochs
        key: jax.random.PRNGKey that has been pre split from a key (jax.random.PRNGKey(42) by default)
        batch_size: data batches defaults to 128
        learning_rate: optimizer learning rate defaults to 0.001
        steps: learning steps defaults to 60000//128=468
        val_steps: validation steps defaults to 10000//128=78
    Returns:
        Trained combined model for saving and running as well as the training embeddings
    """
    model=autoencoder(key=key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        """Computes training loss function
        Args:
            model: autoencoder for training
            x: batch of x training data
            y: batch of y training data
        Returns:
            Loss and grads (grads come from @eqx.filter_value_and_grad)
        """
        pred_y=jax.vmap(model)(x)
        # loss=y*jnp.log(pred_y)+(1-y)*jnp.log(1-pred_y)
        return jnp.mean(jnp.clip(pred_y,0)-pred_y*y+jnp.log1p(jnp.exp(-jnp.abs(pred_y))))# -jnp.nanmean(loss)

    @eqx.filter_jit
    def make_step(model,x,y,opt_state):
        """Iteratively runs training step
        Args:
            model: autoencoder for training
            x: batch of x training data
            y: batch of y training data
            opt_state: current optimizer state
        Returns:
            loss
        """
        loss,grads=compute_loss(model,x,y)
        updates,opt_state=optim.update(grads,opt_state)
        model=eqx.apply_updates(model,updates)
        return loss,model,opt_state
    
    def compute_metrics(model,x,y):
        """Compute validation loss aka metrics
        Args:
            model: takes in post trained model at the end of each epoch
            x: batch of x testing data
            y: batch of y testing data
        Returns:
            losses
        """
        pred_y=jax.vmap(model)(x)
        losses=jnp.mean(jnp.clip(pred_y,0)-pred_y*y+jnp.log1p(jnp.exp(-jnp.abs(pred_y))))#-jnp.nanmean(y*jnp.log(pred_y)+(1-y)*jnp.log(1-pred_y))
        return losses
    
    optim=optax.adam(learning_rate)
    opt_state=optim.init(eqx.filter(model, eqx.is_array))
    
    for epoch in range(epochs):
        bar=trange(steps)
        for i in bar:
            bar.set_description(f"Epoch {epoch+1} of {epochs}")
            start=i*batch_size
            end=start+batch_size

            loss,model,opt_state=make_step(model,x_train[start:end],y_train[start:end],opt_state)
            loss=loss.item()
            bar.set_postfix(loss=loss)

            metrics=[]
            for i in range(val_steps):
                start=i*batch_size
                end=start+batch_size

                metric=compute_metrics(model,x_test[start:end],y_test[start:end])
                metrics.append(metric)

            print(f"Validation Loss: {jnp.nanmean(jnp.array(metrics))}")
        print(f"Epoch {epoch+1} of {epochs} complete")

    trained_encoder=model.modules[0]
    training_embeddings=jax.vmap(trained_encoder)(x_train)
    training_embeddings=jax.device_get(training_embeddings)
    return model, training_embeddings

# ============================================================================ #
# Load models
def loadModels(data_name, model_name, key):
    key, model_key = jax.random.split(key, 2)
    model_template=autoencoder(key=model_key)
    model=eqx.tree_deserialise_leaves(f"/fred/oz440/jonah/module_equinox_cae/results/trained_models/{data_name}_{model_name}.eqx", model_template)
    model=eqx.tree_deserialise_leaves(f"/fred/oz440/jonah/module_equinox_cae/results/trained_models{data_name}_{model_name}.eqx", model_template)
    encoder=model.modules[0]
    decoder=model.modules[1]
    # encoder=eqx.tree_deserialise_leaves(encoder_template, f"/fred/oz440/jonah/equinox_cae/model/{data_name}_{encoder_name}.eqx")
    # decoder=eqx.tree_deserialise_leaves(decoder_template, f"/fred/oz440/jonah/equinox_cae/model/{data_name}_{decoder_name}.eqx")
    return encoder, decoder

def saveModels(data_name, model_name, model):
    eqx.tree_serialise_leaves(f"/fred/oz440/jonah/module_equinox_cae/results/trained_models/{data_name}_{model_name}.eqx",model)
    
def loadData(data_name):
    # The following works for data formatted like tensorflow.keras.datasets.mnist and uses the .npz format
    data=jnp.load(f"/fred/oz440/jonah/data/{data_name}.npz")
    train=data['x_train']
    test=data['x_test']
    return train, test

def preprocess(array):
    array=array.astype("float32")/255.0
    array=jnp.resize(jnp.array(array), (len(array),1,28,28,))
    return array

def noise(array,key,noise_factor=None):
    normal_key,uniform_key=jax.random.split(key,2)
    if noise_factor==None:
        noise_factor=jax.random.uniform(uniform_key,shape=(array.shape[0],1,1,1),minval=0.0,maxval=1.0)
    noisy_array=array+noise_factor*jax.random.normal(normal_key,array.shape)
    return jnp.clip(noisy_array,0.,1.)

def saveImg(non_noisy, noisy, predictions,embeddings, key, plot_name,data_name, n=10):
    """
    Saves ten random image from each one of the supplied arrays
    """
    idx=jax.random.randint(key, (10,),0,len(non_noisy))
    non_noisy_imgs=non_noisy[idx,:]
    noisy_imgs=noisy[idx,:]
    embed_imgs=embeddings[idx,:]
    pred_imgs=predictions[idx,:]

    fig=plt.figure(figsize=(20,6),layout="tight")
    plt.suptitle(f"Images from {data_name} - {plot_name}", fontsize=16)

    for i, (x_img, y_img, embed_img, pred_img) in enumerate(zip(non_noisy_imgs,noisy_imgs,embed_imgs,pred_imgs)):
        ax=plt.subplot(3,n,i+1)
        plt.imshow(y_img.reshape(28,28))
        plt.plasma()
        ax.set_axis_off()
        if i==0:
            ax.set_title("Target Data",loc="left",fontsize=10)
        
        ax=plt.subplot(3,n,i+1+n)
        plt.imshow(x_img.reshape(28,28))
        plt.plasma()
        ax.set_axis_off()
        if i==0:
            ax.set_title("Input Data",loc="left",fontsize=10)
        
        ax=plt.subplot(3,n,i+1+2*n)
        plt.imshow(pred_img.reshape(28,28))
        plt.plasma()
        ax.set_axis_off
        if i==0:
            ax.set_title("Predictions",loc="left",fontsize=10)
        # ax=plt.subplot(4,n,i+1)
        # plt.imshow(y_img.reshape(28,28))
        # plt.plasma()
        # ax.set_axis_off()
        # if i==0:
        #     ax.set_title("Target Data",loc="left",fontsize=10)
        
        # ax=plt.subplot(4,n,i+1+n)
        # plt.imshow(x_img.reshape(28,28))
        # plt.plasma()
        # ax.set_axis_off()
        # if i==0:
        #     ax.set_title("Input Data",loc="left",fontsize=10)

        # ax=plt.subplot(4,n,i+1+2*n)
        # plt.imshow(embed_img.reshape(28,28))
        # plt.plasma()
        # ax.set_axis_off()
        # if i==0:
        #     ax.set_title("Embeddings",loc="left",fontsize=10)
        
        # ax=plt.subplot(4,n,i+1+3*n)
        # plt.imshow(pred_img.reshape(28,28))
        # plt.plasma()
        # ax.set_axis_off
        # if i==0:
        #     ax.set_title("Predictions",loc="left",fontsize=10)
    plt.savefig(f"/fred/oz440/jonah/module_equinox_cae/results/plots/{data_name}_{plot_name}.png")
    plt.close()

def saveData(result, result_type,result_name,data_name):
    file_name=f"/fred/oz440/jonah/module_equinox_cae/results/{result_type}/{data_name}_{result_name}.npz"
    return jnp.savez(file_name, result=result)

# ============================================================================ #
# Running Models
# ============================================================================ #
def runModel(modelPortion,x):
    return jax.vmap(modelPortion)(x)

# ============================================================================ #
# RUN SCRIPT
# ============================================================================ #
if __name__=="__main__":
    args=parse_args()
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")

    primary_key=jax.random.PRNGKey(42)
    model_key,noise_key1,noise_key2,display_key=jax.random.split(primary_key,4)
    
    train_data,test_data=loadData(args.data)
    train_data=jnp.array(preprocess(train_data))
    test_data=jnp.array(preprocess(test_data))

    noisy_train_data=noise(train_data,noise_key1)
    noisy_test_data=noise(test_data,noise_key2)

    n_epochs=args.epochs

    if args.model is not None:
        print("NOT SET UP YET")
    else:
        # print("training non noisy")
        # non_noisy_model, non_noisy_training_embeddings=train(
        #     x_train=train_data,
        #     y_train=train_data,
        #     x_test=test_data,
        #     y_test=test_data,
        #     epochs=n_epochs,
        #     key=model_key
        # )
        # saveData(non_noisy_training_embeddings,"embeddings","non_noisy_train",args.data)
        # saveModels(args.data,"non_noisy",non_noisy_model)
        # non_noisy_encoder=non_noisy_model.modules[0]
        # non_noisy_decoder=non_noisy_model.modules[1]
        # print("running Non Noisy")
        # non_noisy_embeddings=runModel(non_noisy_encoder,train_data).block_until_ready()
        # non_noisy_predictions=runModel(non_noisy_model,noisy_test_data).block_until_ready()
        # saveData(jax.device_get(non_noisy_embeddings),"embeddings","non_noisy",args.data)
        # saveData(jax.device_get(jax.nn.sigmoid(non_noisy_predictions)),"predictions","non_noisy",args.data)
        # saveImg(test_data,test_data,jax.nn.sigmoid(non_noisy_predictions),non_noisy_embeddings,display_key,"non_noisy",args.data)

        print("training noisy")
        noisy_model, noisy_training_embeddings=train(
            x_train=noisy_train_data,
            y_train=train_data,
            x_test=noisy_test_data,
            y_test=test_data,
            epochs=n_epochs,
            key=model_key
        )
        saveData(noisy_training_embeddings,"embeddings","noisy_train",args.data)
        saveModels(args.data,"noisy",noisy_model)
        noisy_encoder=noisy_model.modules[0]
        noisy_decoder=noisy_model.modules[1]
        print("running Noisy")
        noisy_embeddings=runModel(noisy_encoder,noisy_train_data).block_until_ready()
        noisy_predictions=runModel(noisy_model,noisy_test_data).block_until_ready()
        saveData(jax.device_get(noisy_embeddings),"embeddings","noisy",args.data)
        saveData(jax.device_get(jax.nn.sigmoid(noisy_predictions)),"predictions","noisy",args.data)
        saveImg(test_data,noisy_test_data,jax.nn.sigmoid(noisy_predictions),noisy_embeddings,display_key,"noisy",args.data)

        print(":D")