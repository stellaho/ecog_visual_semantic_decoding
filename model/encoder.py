from model.eegnet import EEGNetEncoder
from model.lsnet import LSNetEncoder
from model.lstm import LSTMEncoder
from model.resnet import ResNet1DEncoder
from model.tcn import TCNEncoder
from model.transformer import TransformerEncoder

ENCODERS = ("tcn", "resnet", "lsnet", "lstm", "transformer", "eegnet")


def build_encoder(encoder_type, params, n_channels, n_time, pool_output=True):
   
    if encoder_type == "tcn":
        dilation = (
            params["dilation"]
            if "dilation" in params
            else params.get("dilation_base", 2)
        )
        return TCNEncoder(
            n_inputs=n_channels,
            base_channels=params["base_channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
            n_layers=params["n_layers"],
            dilation_base=dilation,
            pool_output=pool_output,
        )
    
    if encoder_type == "resnet":
        return ResNet1DEncoder(
            n_inputs=n_channels,
            base_channels=params["base_channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
            n_layers=params["n_layers"],
            pool_output=pool_output,
        )
    
    if encoder_type == "lsnet":
        return LSNetEncoder(
            n_inputs=n_channels,
            base_channels=params["base_channels"],
            dropout=params["dropout"],
            n_layers=params["n_layers"],
            pool_output=pool_output,
        )
    
    if encoder_type == "lstm":
        return LSTMEncoder(
            n_inputs=n_channels,
            hidden_dim=params["base_channels"],
            num_layers=params["n_layers"],
            dropout=params["dropout"],
            pool_output=pool_output,
        )
    
    if encoder_type == "transformer":
        return TransformerEncoder(
            n_chl=n_channels,
            n_time=n_time,
            base_channels=params["base_channels"],
            n_layers=params["n_layers"],
            n_heads=params["n_heads"],
            dropout=params["dropout"],
            pos_encoding=params["pos_encoding"],
            pool_output=pool_output,
        )
    
    if encoder_type == "eegnet":
        return EEGNetEncoder(
            n_inputs=n_channels,
            f1=params["eegnet_f1"],
            depth_multiplier=params["eegnet_depth_multiplier"],
            f2=params["eegnet_f2"],
            temporal_kernel_length=params["eegnet_temporal_kernel"],
            separable_kernel_length=params["eegnet_separable_kernel"],
            pool1_kernel=params["eegnet_pool1"],
            pool2_kernel=params["eegnet_pool2"],
            dropout=params["dropout"],
            spatial_max_norm=params["eegnet_spatial_max_norm"],
            pool_output=pool_output,
        )
    raise ValueError(f"Unsupported encoder: {encoder_type}")
