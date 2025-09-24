import argparse
import importlib.util, sys, os
import jax
import faulthandler


from train import train
from io_utils import (
    save_model,
    save_embeddings,
    run_sigmoid,
    save_predictions,
    load_train_val,
    preprocess,
    noise,
    key_handler
)
from models.primary_autoencoder import autoencoder

print("running")


def load_model_factory(file_path: str):
    module_name = "user_model"  # stable name across runs
    file_path = os.path.abspath(file_path)

    # If we've already loaded it, reuse (prevents duplicate identities)
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    # if hasattr(module, "build_autoencoder") and callable(module.build_autoencoder):
    #     return module.build_autoencoder

    # if hasattr(module, "autoencoder") and callable(module.autoencoder):
    #     def _factory(key):
    #         return module.autoencoder(key=key)
    #     return _factory

    raise AttributeError(
        f"{file_path} must define `build_autoencoder(key)` or an `autoencoder` class accepting `key`."
    )

def prep_data(input_path,key,noise_factor=None):
    train_key,test_key=jax.random.split(key,2)
    train_data,test_data=load_train_val(input_path)
    y_train,y_test=preprocess(train_data),preprocess(test_data)
    x_train=noise(y_train,noise_factor,train_key)
    x_test=noise(y_test,noise_factor,test_key)
    return x_train,y_train,x_test,y_test

def run_model(model,x):
    return jax.nn.sigmoid(jax.vmap(model)(x))

def main():
    print("parsing")
    parser=argparse.ArgumentParser()

    #Required paths
    parser.add_argument("-mf","--model_file",required=True, help="Path to model class as .py file")
    parser.add_argument("-df","--data_file",required=True,help="Path to input .npz file")
    parser.add_argument("-fmp","--final_model_path",required=True,help="Where the final model is stored")
    parser.add_argument("-lp","--loss_path",required=True,help="path to where training/validation losses")
    parser.add_argument("-ep","--embeddings_path",required=True, help="Where the training embeddings are saved")

    # optional checkpoints
    parser.add_argument("--resume_from", default=None, help="Resume from a checkpoint file")
    parser.add_argument("--periodic_model_path", default=None, help="Path to save periodic checkpoints")
    parser.add_argument("--best_model_path", default=None, help="Path to save best-val-loss checkpoint")
    parser.add_argument("--checkpoint_every_steps", type=int, default=None, help="Steps between checkpoints")

    # other parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=None, help="Steps per epoch (default: len(x_train)//batch_size)")
    parser.add_argument("--val_steps", type=int, default=None, help="Validation steps per epoch (default: len(x_test)//batch_size)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise_factor", type=float, default=None, help="noise factor as float. By default it is None which sets the noise factor to a Uniform distribution between 0 and 1")
    parser.add_argument("--clean_run", type=bool, default=False, help="additionally trains model on non noisy data")

    args=parser.parse_args()
    primary_key, model_key, noise_key, display_key=key_handler(args.seed)

    if args.clean_run==True:
        model_key, non_noise_model_key=jax.random.split(model_key,2)

    x_train,y_train,x_test,y_test=prep_data(args.data_file,noise_key)

    trained_model, training_embeddings, train_losses, val_losses=train(
        model_key,
        x_train, y_train, x_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        steps=args.steps,
        val_steps=args.val_steps,
        autoencoder=autoencoder,
        loss_path=args.loss_path,
        embeddings_path=args.embeddings_path,
        final_model_path=args.final_model_path,
        resume_from=args.resume_from,
        checkpoint_every_steps=args.checkpoint_every_steps,
        periodic_model_path=args.periodic_model_path,
        best_model_path=args.best_model_path
    )

if __name__ == "__main__":
    main()