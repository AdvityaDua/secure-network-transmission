import sys
import os
import torch
from torchview import draw_graph

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.res_ae import ResAE
from models.res_cbam_xl import ResAutoencoderXLCBAM
from models.unet_one import UNet as UNetOne
from models.unet_no_skip import UNet as UNetNoSkip

def main():
    out_dir = 'results/arch_diagrams'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating diagram for res_ae...")
    model1 = ResAE(in_channels=1, base_channels=32)
    draw_graph(model1, input_size=(1, 1, 256, 256), expand_nested=True, save_graph=True, filename='res_ae', directory=out_dir)
    
    print("Generating diagram for res_cbam_xl...")
    model2 = ResAutoencoderXLCBAM(base_channels=32)
    draw_graph(model2, input_size=(1, 1, 256, 256), expand_nested=True, save_graph=True, filename='res_cbam_xl', directory=out_dir)
    
    print("Generating diagram for unet_one...")
    model3 = UNetOne()
    draw_graph(model3, input_size=(1, 1, 256, 256), expand_nested=True, save_graph=True, filename='unet_one', directory=out_dir)
    
    print("Generating diagram for unet_no_skip...")
    model4 = UNetNoSkip()
    draw_graph(model4, input_size=(1, 1, 256, 256), expand_nested=True, save_graph=True, filename='unet_no_skip', directory=out_dir)
    
    print("Done! Check results/arch_diagrams for PNG files.")

if __name__ == "__main__":
    main()
