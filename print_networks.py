import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

LAYOUT = 'horizontal'

# Configurations
ENV_CONFIGS = {
    'smc_discrete':   {
        'file': 'champions/smc_best.out', 
        'n_in': 2, 'n_out': 3,
        'labels': ['position', 'velocity', 'accelerate\nleft', 'no\nacceleration', 'accelerate\nright']
    },
    'smc_continuous': {
        'file': 'champions/smc_conti_best.out', 
        'n_in': 2, 'n_out': 1,
        'labels': ['position', 'velocity', 'acceleration']
    },
    'lunar_lander':   {
        'file': 'champions/lula_best.out', 
        'n_in': 8, 'n_out': 4,
        'labels': ['x', 'y', 'dx', 'dy', 'angle', 'angular\nvel', 'leg1', 'leg2',
                   'do\nnothing', 'fire\nleft', 'fire\nmain', 'fire\nright']
    },
}

ACT_SHORT_LABELS = {
    1: '(lin)', 2: '(0/1)', 3: '(sin)', 4: '(gaus)', 5: '(tanh)',
    6: '(sig)', 7: '(inv)', 8: '(abs)', 9: '(relu)', 10: '(cos)', 11: '(x^2)'
}

def cLinspace(start, end, N):
    if N == 1:
        return np.mean([start, end])
    else:
        return np.linspace(start, end, N)

def getLayer(wMat):
    wMat = np.copy(wMat)
    wMat[np.isnan(wMat)] = 0
    wMat[wMat != 0] = 1
    nNode = np.shape(wMat)[0]
    layer = np.zeros((nNode))
    while True:
        prevOrder = np.copy(layer)
        for curr in range(nNode):
            srcLayer = np.zeros((nNode))
            for src in range(nNode):
                srcLayer[src] = layer[src] * wMat[src, curr]
            layer[curr] = np.max(srcLayer) + 1
        if np.all(prevOrder == layer):
            break
    return layer - 1

def ind2graph(wMat, nIn, nOut):
    hMat = wMat[nIn:-nOut, nIn:-nOut]
    hLay = getLayer(hMat) + 1

    if len(hLay) > 0:
        lastLayer = np.max(hLay) + 1
    else:
        lastLayer = 1
    L = np.r_[np.zeros(nIn), hLay, np.full((nOut), lastLayer)]

    layer = L
    order = layer.argsort()
    layer = layer[order]

    wMat_sorted = wMat[np.ix_(order, order)]

    rows, cols = np.where(~np.isnan(wMat_sorted) & (wMat_sorted != 0))
    edges = list(zip(rows.tolist(), cols.tolist()))
    G = nx.DiGraph()
    G.add_nodes_from(range(len(layer)))
    G.add_edges_from(edges)
    return G, layer, order, wMat_sorted

def getNodeCoord(G, layer, layout='horizontal'):
    nNode = len(G.nodes)
    if nNode == 0:
        return {}
    fig_wide = 10
    fig_long = 5
    
    x = np.ones((1, nNode)) * layer
    if np.max(x) > 0:
        x = (x / np.max(x)) * fig_wide

    _, nPerLayer = np.unique(layer, return_counts=True)

    y = cLinspace(-2, fig_long + 2, nPerLayer[0])
    for i in range(1, len(nPerLayer)):
        if i % 2 == 0:
            y = np.r_[y, cLinspace(0, fig_long, nPerLayer[i])]
        else:
            y = np.r_[y, cLinspace(-1, fig_long + 1, nPerLayer[i])]

    if layout == 'vertical':
        # Swap X and Y, and invert the new Y so inputs (layer 0) are at the top
        pos = dict(enumerate(np.c_[y.T, -x.T].tolist()))
    else:
        pos = dict(enumerate(np.c_[x.T, y.T].tolist()))
    return pos

def plot_network(name, config_dict, output_dir):
    weight_file = config_dict['file']
    nIn = config_dict['n_in'] + 1 
    nOut = config_dict['n_out']
    labels = config_dict['labels']
    
    try:
        data = np.loadtxt(weight_file, delimiter=',')
    except OSError:
        print(f"File not found: {weight_file}")
        return
        
    wMat = data[:, :-1]
    aVec = data[:, -1]
    wMat[np.isclose(wMat, 0)] = np.nan
    
    G, layer, order, wMat_sorted = ind2graph(wMat, nIn, nOut)
    pos = getNodeCoord(G, layer, layout=LAYOUT)
    aVec_sorted = aVec[order]
    
    # Force exactly the same absolute figure size for all three networks
    fig_width = 18 
    fig_height = 12

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
    ax = fig.add_subplot(111)
    
    # GRAPHICS VARIABLES
    node_size = 3500          
    font_size = 21            
    edge_width = 3.5          
    arrow_size = 30           
    
    nx.draw_networkx_edges(G, pos,
                           alpha=0.6, width=edge_width, 
                           edge_color='gray', arrowsize=arrow_size)
    
    cmap = plt.cm.Set3 
    unique_types = np.unique(aVec_sorted)
    color_mapping = {atype: cmap(i / max(1, len(unique_types))) for i, atype in enumerate(unique_types)}
    node_colors = [color_mapping[atype] for atype in aVec_sorted]

    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors, edgecolors='black', 
                           node_shape='o', node_size=node_size)
                           
    node_labels = {}
    for i in range(len(G.nodes)):
        act_id = int(aVec_sorted[i])
        act_str = ACT_SHORT_LABELS.get(act_id, str(act_id))
        node_labels[i] = f"{act_str}"
        
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size)
    
    stateLabels = ['bias'] + labels
    labelDict = {}
    nNode = len(G.nodes)
    fixed_nodes = np.r_[np.arange(0, nIn), np.arange(nNode - nOut, nNode)]
    
    for i in range(len(stateLabels)):
        if i < len(fixed_nodes):
            labelDict[fixed_nodes[i]] = stateLabels[i]

    # Draw inputs
    for i in range(nIn):
        if i in pos:
            if LAYOUT == 'vertical':
                plt.annotate(labelDict[i], 
                             xy=(pos[i][0], pos[i][1] + 0.2), 
                             xytext=(pos[i][0], pos[i][1] + 1.2), 
                             arrowprops=dict(arrowstyle="->", color='black', shrinkB=np.sqrt(node_size)/1.5, connectionstyle="arc3"),
                             fontsize=28, fontweight='bold', ha='center', va='bottom')
            else:
                plt.annotate(labelDict[i], 
                             xy=(pos[i][0] - 0.2, pos[i][1]), 
                             xytext=(pos[i][0] - 1.5, pos[i][1]), 
                             arrowprops=dict(arrowstyle="->", color='black', shrinkB=np.sqrt(node_size)/1.5, connectionstyle="arc3"),
                             fontsize=28, fontweight='bold', ha='right', va='center')

    # Draw outputs
    for i in range(nNode - nOut, nNode):
        if i in pos:
            if LAYOUT == 'vertical':
                plt.annotate(labelDict[i], 
                             xy=(pos[i][0], pos[i][1] - 0.2), 
                             xytext=(pos[i][0], pos[i][1] - 1.2),
                             arrowprops=dict(arrowstyle="<-", color='black', shrinkB=np.sqrt(node_size)/1.5, connectionstyle="arc3"),
                             fontsize=28, fontweight='bold', ha='center', va='top')
            else:
                plt.annotate(labelDict[i], 
                             xy=(pos[i][0] + 0.2, pos[i][1]), 
                             xytext=(pos[i][0] + 1.5, pos[i][1]),
                             arrowprops=dict(arrowstyle="<-", color='black', shrinkB=np.sqrt(node_size)/1.5, connectionstyle="arc3"),
                             fontsize=28, fontweight='bold', ha='left', va='center')
                         
    plt.axis('off')
    
    out_path = os.path.join(output_dir, f"{name}.pdf")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close()

if __name__ == '__main__':
    out_dir = "network_images"
    os.makedirs(out_dir, exist_ok=True)
    for name, config_dict in ENV_CONFIGS.items():
        plot_network(name, config_dict, out_dir)
