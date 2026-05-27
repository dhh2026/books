
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import graph_tool.all as gt
import scipy.sparse as sparse
from metrics_helpers import gini
import networkx as nx
from networkx import bipartite

city = 'wittenberg'

def next_time_window(result_dict):
    statuses = [v['status'] for v in result_dict.values()]
    time_windows = [t for t in result_dict.keys()]

    i = 0
    while i < len(statuses):
        if statuses[i] == None:
            break
        else:
            i += 1

    start, end = time_windows[i].split('-')
    start, end = int(start), int(end)
    return start, end

def get_distance_stats(lcc: gt.Graph,
                       # adj_mat: sparse.csr_matrix, 
                       # v_module: gt.VertexPropertyMap, 
                       statistic_func=np.mean
                       ):
    # g = gt.Graph(adj_mat)
    # g.vp['v_module'] = g.new_vp('int')
    # g.vp['v_module'].a = v_module.a
    # lcc = gt.extract_largest_component(g, prune=True)
    v_module_map = lcc.vp['v_type']
    v_publisher = np.argwhere(v_module_map.a==1).ravel()
    v_author = np.argwhere(v_module_map.a==0).ravel()
    dists = []
    for v in v_publisher:
        dist_map = gt.shortest_distance(lcc, source=v, target=v_author)
        dists += dist_map.tolist()
    return statistic_func(dists)

# def bipartite_clustering_coef(g):
#     clust = gt.extended_clustering(g, max_depth=4, undirected=True)
#     avg_clust, _ = gt.vertex_average(g, clust[3])
#     return avg_clust*4*3*2

def to_networkx_bipartite(g: gt.Graph, 
                          node_names: str,
                          node_types: str,
                          weight: str
                          ):
    vmap = g.vp[node_names]
    vtype= g.vp[node_types]
    weights = g.ep[weight]
    upper_nodes = [vmap[v] for v in np.where(vtype.a==1)[0].tolist()]
    lower_nodes = [vmap[v] for v in np.where(vtype.a==0)[0].tolist()]
    ebunchs = [(vmap[e.source()], vmap[e.target()], weights[e]) for e in g.edges()]
    
    nx_bg = nx.Graph()
    nx_bg.add_nodes_from(upper_nodes, bipartite=1)
    nx_bg.add_nodes_from(lower_nodes, bipartite=0)
    nx_bg.add_weighted_edges_from(ebunchs)

    return nx_bg

if __name__=='__main__':

    print('\nLoading data...')
    
    with open(f'../../data/work/{city}_stats.json', 'r') as f:
        all_results = json.load(f)
    # print(all_results)
    start_time, end_time = next_time_window(all_results)
    window = f'{start_time}-{end_time}'

    # load the data for use
    df = pd.read_json(f'../../data/work/vd_all_{city}.jsonl', lines=True, orient='records')

    elist = df[df.year.between(start_time, end_time)]
    elist = elist.groupby(by=["publisher", "author"])["year"].nunique().reset_index()

    print(f'\nBuilding the network snapshot for {window}...')
    # build the graph
    g = gt.Graph(directed=False)
    vmap = g.add_edge_list(list(zip(elist["publisher"], 
                            elist["author"], 
                            elist['year'])
                            ), 
                            hashed=True, 
                            eprops=[('weight', 'double')])

    vtypes = {n:1 for n in elist.publisher}
    vtypes.update({n:0 for n in elist.author})
    vtype = g.new_vertex_property('int')
    vtype.a = list(map(lambda x: vtypes.get(x), list(vmap)))
    g.vertex_properties['v_type'] = vtype
    g.vertex_properties['name'] = vmap

    print('\nComputing stats...')
    # adjacency matrices of configuration model networks
    rand_gs = []
    niter = 20
    for i in tqdm(range(niter)):
        rand_g = g.copy()
        gt.random_rewire(rand_g, 
                        # weight option to be added
                        model="constrained-configuration", 
                        block_membership=rand_g.vp['v_type'])
        this_adj = gt.adjacency(rand_g, #weight=rand_g.ep.weight
                                ).copy()
        rand_gs.append(this_adj)

    
    # get degrees gini coef
    deg = g.degree_property_map(deg='total')
    upper_degrees = deg.a[np.where(vtype.a == 1)]
    lower_degrees = deg.a[np.where(vtype.a == 0)]

    all_results[window]['updeg_gini'] = float(gini(upper_degrees))
    all_results[window]['lowdeg_gini'] = float(gini(lower_degrees))

    # get clustering coefs
    nx_bg = to_networkx_bipartite(g, 'name', 'v_type', 'weight')
    obs_clustering = bipartite.average_clustering(nx_bg)
    
    # obs_clustering = bipartite_clustering_coef(g)
    lcc = gt.extract_largest_component(g, prune=True)
    obs_dist = get_distance_stats(lcc)

    rand_avg_clusterings = []
    rand_dists = []
    for mat in tqdm(rand_gs):
        rand_g = gt.Graph(mat)
        rand_g.vp['v_type'] = rand_g.new_vp('int')
        rand_g.vp['v_type'].a = vtype.a
        lcc = gt.extract_largest_component(rand_g, prune=True)
        rand_dists.append(get_distance_stats(lcc))
        # print(rand_g.vp['v_module'].a)

        # lcc = gt.extract_largest_component(rand_g, prune=True)
        # print(lcc.vp['v_module'].a)
        # rand_g.vp['v_type'] = rand_g.new_vp('int')
        # rand_g.vp['v_type'].a = vtype.a

        rand_vmap = rand_g.new_vp('string')
        for i in range(len(rand_vmap)):
            rand_vmap[i] = vmap[i]
        rand_g.vp['name'] = rand_vmap
        # rand_nxbg = to_networkx_bipartite(rand_g, 'name', 'v_type', 'weight')
        # rand_avg_clusterings.append(bipartite.average_clustering(rand_nxbg))
        rand_nxbg = to_networkx_bipartite(rand_g, 'name', 'v_type', 'weight')
        rand_avg_clusterings.append(bipartite.average_clustering(rand_nxbg))
        
    
    estimates = obs_clustering/np.array(rand_avg_clusterings)
    all_results[window]['clustering']['obs'] = obs_clustering
    all_results[window]['clustering']['coef_mean'] = np.mean(estimates)
    all_results[window]['clustering']['coef_lb'] = np.quantile(estimates, q=.05)
    all_results[window]['clustering']['coef_ub'] = np.quantile(estimates, q=.95)
    
    estimates = obs_dist/np.array(rand_dists)
    all_results[window]['distance']['obs'] = obs_dist
    all_results[window]['distance']['coef_mean'] = np.mean(estimates)
    all_results[window]['distance']['coef_lb'] = np.quantile(estimates, q=.05)
    all_results[window]['distance']['coef_ub'] = np.quantile(estimates, q=.95)

    print('\nFinished computing. Saving results...')
    all_results[window]['status'] = 'completed'

    # Save to JSON file
    with open(f'../../data/work/{city}_stats.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\nResults saved.')