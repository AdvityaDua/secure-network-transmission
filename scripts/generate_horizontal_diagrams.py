import graphviz
import os

out_dir = 'results/arch_diagrams_horizontal'
os.makedirs(out_dir, exist_ok=True)

def create_ae_graph(name, title, block_name, has_skips=False):
    dot = graphviz.Digraph(name, format='png')
    
    # Make it strictly Left-to-Right
    dot.attr(rankdir='LR')
    # Set reasonable node styling
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Helvetica', margin='0.1')
    dot.attr('edge', fontname='Helvetica')
    
    # Input
    dot.node('Input', 'Input Image', shape='ellipse', fillcolor='lightgrey')
    
    # Main path
    prev_node = 'Input'
    
    # Encoder
    for i in range(1, 5):
        enc_name = f'Enc{i}'
        dot.node(enc_name, f'{block_name}\nEnc {i}', fillcolor='lightblue')
        dot.edge(prev_node, enc_name)
        
        pool_name = f'Pool{i}'
        dot.node(pool_name, 'MaxPool', fillcolor='lightyellow')
        dot.edge(enc_name, pool_name)
        prev_node = pool_name
        
    # Bottleneck
    dot.node('Bottleneck', f'Bottleneck\n({block_name})', fillcolor='lightcoral')
    dot.edge(prev_node, 'Bottleneck')
    prev_node = 'Bottleneck'
    
    # Decoder
    for i in range(4, 0, -1):
        up_name = f'Up{i}'
        dot.node(up_name, 'UpConv', fillcolor='lightyellow')
        dot.edge(prev_node, up_name)
        
        dec_name = f'Dec{i}'
        if has_skips:
            dot.node(dec_name, f'Concat +\n{block_name}\nDec {i}', fillcolor='lightgreen')
        else:
            dot.node(dec_name, f'{block_name}\nDec {i}', fillcolor='lightgreen')
            
        dot.edge(up_name, dec_name)
        prev_node = dec_name
        
    # Output
    dot.node('Output', 'Output Image', shape='ellipse', fillcolor='lightgrey')
    dot.edge(prev_node, 'Output')
    
    # Skip connections
    if has_skips:
        for i in range(1, 5):
            # Using constraint=false prevents the skip connections from messing up the LR layout
            dot.edge(f'Enc{i}', f'Dec{i}', label='Skip', style='dashed', constraint='false', color='gray50')
            
    dot.render(os.path.join(out_dir, name))
    print(f"Generated {name}.png")

def main():
    create_ae_graph('unet_no_skip_horizontal', 'UNet (No Skip)', 'DoubleConv', False)
    create_ae_graph('unet_one_horizontal', 'UNet (With Skips)', 'DoubleConv', True)
    create_ae_graph('res_ae_horizontal', 'ResNet Autoencoder', 'ResBlock', False)
    create_ae_graph('res_cbam_xl_horizontal', 'ResNet CBAM XL', 'CBAM_ResBlock', False)
    
if __name__ == "__main__":
    main()
