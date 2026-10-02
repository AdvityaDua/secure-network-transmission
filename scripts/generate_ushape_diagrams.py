import graphviz
import os

out_dir = 'results/arch_diagrams_ushape'
os.makedirs(out_dir, exist_ok=True)

def create_ushape_graph(name, title, block_name, has_skips=False):
    dot = graphviz.Digraph(name, format='png')
    
    # Left-to-Right ranking
    dot.attr(rankdir='LR', ranksep='0.5', nodesep='0.5', splines='ortho')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Helvetica', margin='0.1')
    dot.attr('edge', fontname='Helvetica', penwidth='1.5')
    
    # -----------------------------
    # Define Nodes
    # -----------------------------
    dot.node('Input', 'Input Image', shape='ellipse', fillcolor='lightgrey')
    dot.node('Output', 'Output Image', shape='ellipse', fillcolor='lightgrey')
    
    for i in range(1, 5):
        dot.node(f'Enc{i}', f'{block_name}\nEnc {i}', fillcolor='lightblue')
        dot.node(f'Pool{i}', 'MaxPool\n2x2', fillcolor='lightyellow')
        
        dot.node(f'Up{i}', 'UpConv\n2x2', fillcolor='lightyellow')
        if has_skips:
            dot.node(f'Dec{i}', f'Concat +\n{block_name}\nDec {i}', fillcolor='lightgreen')
        else:
            dot.node(f'Dec{i}', f'{block_name}\nDec {i}', fillcolor='lightgreen')
            
    dot.node('Bottleneck', f'Bottleneck\n({block_name})', fillcolor='lightcoral')
    
    # -----------------------------
    # Define Edges (Data Flow)
    # -----------------------------
    # Encoder Path (Left to Right)
    dot.edge('Input', 'Enc1')
    dot.edge('Enc1', 'Pool1')
    dot.edge('Pool1', 'Enc2')
    dot.edge('Enc2', 'Pool2')
    dot.edge('Pool2', 'Enc3')
    dot.edge('Enc3', 'Pool3')
    dot.edge('Pool3', 'Enc4')
    dot.edge('Enc4', 'Pool4')
    
    # Down to Bottleneck
    dot.edge('Pool4', 'Bottleneck')
    
    # Decoder Path (Right to Left)
    dot.edge('Bottleneck', 'Up4')
    dot.edge('Up4', 'Dec4')
    dot.edge('Dec4', 'Up3')
    dot.edge('Up3', 'Dec3')
    dot.edge('Dec3', 'Up2')
    dot.edge('Up2', 'Dec2')
    dot.edge('Dec2', 'Up1')
    dot.edge('Up1', 'Dec1')
    dot.edge('Dec1', 'Output')
    
    # Skip connections (Top to Bottom)
    if has_skips:
        for i in range(1, 5):
            dot.edge(f'Enc{i}', f'Dec{i}', label='Skip', style='dashed', color='gray40', constraint='false')
            
    # -----------------------------
    # Enforce Ranks (U-Shape Alignment)
    # -----------------------------
    # Force alignment and top-to-bottom relative ordering via invisible edges
    alignments = [
        ('Input', 'Output'),
        ('Enc1', 'Dec1'),
        ('Pool1', 'Up1'),
        ('Enc2', 'Dec2'),
        ('Pool2', 'Up2'),
        ('Enc3', 'Dec3'),
        ('Pool3', 'Up3'),
        ('Enc4', 'Dec4'),
        ('Pool4', 'Up4'),
    ]
    
    for top, bot in alignments:
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node(top)
            s.node(bot)
        # Invisible edge enforces Top -> Bottom placement
        dot.edge(top, bot, style='invis')
        
    dot.render(os.path.join(out_dir, name))
    print(f"Generated {name}.png")

def main():
    create_ushape_graph('unet_no_skip_ushape', 'UNet (No Skip)', 'DoubleConv', False)
    create_ushape_graph('unet_one_ushape', 'UNet (With Skips)', 'DoubleConv', True)
    create_ushape_graph('res_ae_ushape', 'ResNet Autoencoder', 'ResBlock', False)
    create_ushape_graph('res_cbam_xl_ushape', 'ResNet CBAM XL', 'CBAM_ResBlock', False)
    
if __name__ == "__main__":
    main()
