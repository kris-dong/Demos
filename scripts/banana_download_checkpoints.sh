mkdir -p ./checkpoints

wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true -O ./checkpoints/depth_anything_v2_vits.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vitb.pth?download=true -O ./checkpoints/depth_anything_v2_vitb.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vitl.pth?download=true -O ./checkpoints/depth_anything_v2_vitl.pth

wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true -O ./checkpoints/depth_anything_v2_metric_hypersim_vits.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true -O ./checkpoints/depth_anything_v2_metric_hypersim_vitb.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true -O ./checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
