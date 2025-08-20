import jax
import jax.numpy as jnp
import equinox as eqx


__all__=["encoder","decoder","autoencoder"]

# ============================================================================ #
# Encode
class encoder(eqx.Module):
    """Encoder module for autoencoder

    Args:
        eqx.Module: eequinox base class

    Returns:
        x: returns layer by layer encoder
    """

    encoder_layers: list

    def __init__(self, *, key):
        key1, key2 = jax.random.split(key, 2)
        self.encoder_layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, key=key1),
            jax.nn.relu,
            eqx.nn.MaxPool2d((2, 2), 2),
            eqx.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, key=key2),
            jax.nn.relu,
            eqx.nn.MaxPool2d((2, 2), 2)
        ]
    def __call__(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

# ============================================================================ #
# Decode
class decoder(eqx.Module):
    """Decoder module for autoencoder

    Args:
        eqx.Module: Equinox base class

    Returns:
        x: layer by layer decoder
    """

    decoder_layers: list
    
    def __init__(self, *, key):
        _, _, key3, key4, key5 = jax.random.split(key,5)
        self.decoder_layers = [
            eqx.nn.ConvTranspose(2, in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, key=key3),
            jax.nn.relu,
            eqx.nn.ConvTranspose(2, in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, key=key4),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, key=key5),
            jax.nn.sigmoid
        ]
    def __call__(self, x):
        for layer in self.decoder_layers:
            x = layer(x)
        return x

# ============================================================================ #
# Combined
class autoencoder(eqx.Module):
    """Combined autoencoder pulls in encoder and decoders from above classes

    Args:
        eqx (Module): base equinox class

    Returns:
        x: loops from encoder to decoder, each will go through their own layers
    """
    
    modules: list
    def __init__(self, *, key):
        encoder = autoencoder_encode(key=key)
        decoder = autoencoder_decode(key=key)
        self.modules = [encoder, decoder]
    def __call__(self, x):
        for layer in self.modules:
            x = layer(x)
        return x

