"""
Cycle Feature Analyzer

Analyzes extracted cycle subgraphs to find discriminative features
between trojaned and clean circuits.

This script:
1. Loads extracted cycle data
2. Computes statistical differences between classes
3. Generates new discriminative features
4. Creates a feature importance ranking
5. Outputs enhanced features for training
"""

import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Feature names for the 26-dimensional node features
FEATURE_NAMES = [
    # One-hot gate types (0-13)
    'gate_input', 'gate_output', 'gate_wire', 'gate_and', 'gate_nand',
    'gate_or', 'gate_nor', 'gate_xor', 'gate_xnor', 'gate_not',
    'gate_buf', 'gate_dff', 'gate_mux', 'gate_unknown',
    # Structural features (14-17)
    'fan_in_norm', 'fan_out_norm', 'depth_norm', 'in_cycle_original',
    # Cycle membership (18)
    'in_cycle_subgraph',
    # Neighborhood features (19-25)
    'avg_neighbor_fanin', 'avg_neighbor_fanout', 'inverter_density',
    'sequential_density', 'rare_gate_density', 'neighbor_count', 'self_loop'
]


# =============================================================================
# Data Loading
# =============================================================================

def load_cycle_data(json_path: str) -> Tuple[List[dict], List[dict]]:
    """Load cycle data and split by label."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trojaned = [d for d in data if d['label'] == 1]
    clean = [d for d in data if d['label'] == 0]
    
    print(f"Loaded {len(data)} graphs: {len(trojaned)} trojaned, {len(clean)} clean")
    return trojaned, clean


# =============================================================================
# Statistical Analysis
# =============================================================================

def build_digraph_from_edge_index(edge_index: List[List[int]], num_nodes: int) -> nx.DiGraph:
    """Build a NetworkX DiGraph from edge index."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if len(edge_index) == 2 and len(edge_index[0]) > 0:
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
    return G


def compute_fanin_fanout_cones(G: nx.DiGraph, sample_size: int = 50) -> Dict[str, float]:
    """
    Compute fanin/fanout cone features - critical for circuit analysis.
    
    Fanin cone: all nodes that can reach a given node (predecessors)
    Fanout cone: all nodes reachable from a given node (successors)
    
    Trojans often have unusual cone characteristics:
    - Small fanin cones (few inputs trigger them)
    - Small fanout cones (limited payload reach)
    - Asymmetric cone ratios
    """
    features = {}
    num_nodes = G.number_of_nodes()
    
    if num_nodes == 0:
        return {
            'avg_fanin_cone_size': 0, 'avg_fanout_cone_size': 0,
            'max_fanin_cone_size': 0, 'max_fanout_cone_size': 0,
            'fanin_cone_std': 0, 'fanout_cone_std': 0,
            'cone_ratio_mean': 0, 'cone_ratio_std': 0,
            'small_fanin_cone_fraction': 0, 'small_fanout_cone_fraction': 0,
            'isolated_cone_nodes': 0, 'reconvergent_node_fraction': 0
        }
    
    # Sample nodes for large graphs
    nodes = list(G.nodes())
    if len(nodes) > sample_size:
        sample_nodes = np.random.choice(nodes, sample_size, replace=False)
    else:
        sample_nodes = nodes
    
    fanin_sizes = []
    fanout_sizes = []
    cone_ratios = []
    
    for node in sample_nodes:
        # Fanin cone (ancestors)
        fanin_cone = nx.ancestors(G, node)
        fanin_sizes.append(len(fanin_cone))
        
        # Fanout cone (descendants)
        fanout_cone = nx.descendants(G, node)
        fanout_sizes.append(len(fanout_cone))
        
        # Cone ratio (fanout/fanin) - trojans often have unusual ratios
        if len(fanin_cone) > 0:
            cone_ratios.append(len(fanout_cone) / len(fanin_cone))
        else:
            cone_ratios.append(len(fanout_cone) if len(fanout_cone) > 0 else 0)
    
    fanin_sizes = np.array(fanin_sizes)
    fanout_sizes = np.array(fanout_sizes)
    cone_ratios = np.array(cone_ratios)
    
    features['avg_fanin_cone_size'] = np.mean(fanin_sizes) / max(num_nodes, 1)
    features['avg_fanout_cone_size'] = np.mean(fanout_sizes) / max(num_nodes, 1)
    features['max_fanin_cone_size'] = np.max(fanin_sizes) / max(num_nodes, 1)
    features['max_fanout_cone_size'] = np.max(fanout_sizes) / max(num_nodes, 1)
    features['fanin_cone_std'] = np.std(fanin_sizes) / max(num_nodes, 1)
    features['fanout_cone_std'] = np.std(fanout_sizes) / max(num_nodes, 1)
    features['cone_ratio_mean'] = np.mean(cone_ratios)
    features['cone_ratio_std'] = np.std(cone_ratios)
    
    # Fraction of nodes with small cones (potential trojan indicators)
    threshold = 0.1 * num_nodes
    features['small_fanin_cone_fraction'] = np.sum(fanin_sizes < threshold) / len(sample_nodes)
    features['small_fanout_cone_fraction'] = np.sum(fanout_sizes < threshold) / len(sample_nodes)
    
    # Isolated cone nodes (small both fanin and fanout)
    features['isolated_cone_nodes'] = np.sum((fanin_sizes < threshold) & (fanout_sizes < threshold)) / len(sample_nodes)
    
    # Reconvergent paths detection (nodes reachable by multiple paths)
    reconvergent_count = 0
    for node in sample_nodes[:min(20, len(sample_nodes))]:
        predecessors = list(G.predecessors(node))
        if len(predecessors) >= 2:
            # Check if any two predecessors share ancestors
            for i, p1 in enumerate(predecessors[:5]):
                ancestors1 = nx.ancestors(G, p1)
                for p2 in predecessors[i+1:min(i+6, len(predecessors))]:
                    ancestors2 = nx.ancestors(G, p2)
                    if ancestors1 & ancestors2:  # Shared ancestors = reconvergence
                        reconvergent_count += 1
                        break
    features['reconvergent_node_fraction'] = reconvergent_count / max(len(sample_nodes), 1)
    
    return features


def compute_topological_features(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute topological ordering features for directed graphs.
    
    These capture the hierarchical/layered structure of circuits:
    - Level distribution (gates at each depth level)
    - Cross-level connections (how information flows between levels)
    - Backward edges (violations of topological order = feedback)
    """
    features = {}
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        return {
            'num_levels': 0, 'level_std': 0, 'max_level_width': 0,
            'level_width_std': 0, 'cross_level_edge_fraction': 0,
            'backward_edge_fraction': 0, 'forward_skip_mean': 0,
            'bottleneck_ratio': 0
        }
    
    # Compute levels using BFS from sources
    in_degrees = dict(G.in_degree())
    sources = [n for n, d in in_degrees.items() if d == 0]
    
    if not sources:
        # No sources (all cycles), use any node
        sources = [list(G.nodes())[0]]
    
    # BFS to assign levels
    levels = {}
    for source in sources:
        queue = [(source, 0)]
        while queue:
            node, level = queue.pop(0)
            if node not in levels or levels[node] > level:
                levels[node] = level
                for succ in G.successors(node):
                    queue.append((succ, level + 1))
    
    # Assign remaining nodes (in cycles) maximum level + 1
    max_level = max(levels.values()) if levels else 0
    for node in G.nodes():
        if node not in levels:
            levels[node] = max_level + 1
    
    level_values = list(levels.values())
    num_levels = max(level_values) + 1 if level_values else 0
    
    features['num_levels'] = num_levels / max(num_nodes, 1)
    features['level_std'] = np.std(level_values) / max(num_levels, 1)
    
    # Level width distribution
    level_widths = defaultdict(int)
    for level in level_values:
        level_widths[level] += 1
    
    widths = list(level_widths.values())
    features['max_level_width'] = max(widths) / max(num_nodes, 1) if widths else 0
    features['level_width_std'] = np.std(widths) / max(np.mean(widths), 1) if widths else 0
    
    # Bottleneck ratio (min width / max width) - trojans may create bottlenecks
    features['bottleneck_ratio'] = min(widths) / max(max(widths), 1) if widths else 0
    
    # Edge level analysis
    cross_level_edges = 0
    backward_edges = 0
    forward_skips = []
    
    for src, dst in G.edges():
        src_level = levels.get(src, 0)
        dst_level = levels.get(dst, 0)
        level_diff = dst_level - src_level
        
        if level_diff > 1:
            cross_level_edges += 1
            forward_skips.append(level_diff)
        elif level_diff <= 0:
            backward_edges += 1
    
    features['cross_level_edge_fraction'] = cross_level_edges / max(num_edges, 1)
    features['backward_edge_fraction'] = backward_edges / max(num_edges, 1)
    features['forward_skip_mean'] = np.mean(forward_skips) if forward_skips else 0
    
    return features


def compute_controllability_observability(G: nx.DiGraph, sample_size: int = 100) -> Dict[str, float]:
    """
    Compute simplified SCOAP-like controllability/observability estimates.
    
    Controllability: How easy is it to set a node to 0 or 1 from inputs
    Observability: How easy is it to observe a node's value at outputs
    
    Trojans often have:
    - Low controllability (hard to trigger)
    - Low observability (hard to detect)
    """
    features = {}
    num_nodes = G.number_of_nodes()
    
    if num_nodes == 0:
        return {
            'avg_controllability': 0, 'min_controllability': 0,
            'controllability_std': 0, 'avg_observability': 0,
            'min_observability': 0, 'observability_std': 0,
            'low_ctrl_obs_fraction': 0, 'ctrl_obs_product_min': 0
        }
    
    # Find primary inputs (sources) and primary outputs (sinks)
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    sources = set(n for n, d in in_degrees.items() if d == 0)
    sinks = set(n for n, d in out_degrees.items() if d == 0)
    
    # If no sources/sinks, use heuristics
    if not sources:
        sources = set(list(G.nodes())[:max(1, num_nodes // 10)])
    if not sinks:
        sinks = set(list(G.nodes())[-max(1, num_nodes // 10):])
    
    # Sample nodes
    nodes = list(G.nodes())
    if len(nodes) > sample_size:
        sample_nodes = np.random.choice(nodes, sample_size, replace=False)
    else:
        sample_nodes = nodes
    
    controllabilities = []
    observabilities = []
    
    for node in sample_nodes:
        # Controllability: inverse of shortest path from any source
        ctrl = 0
        for source in list(sources)[:10]:  # Limit for speed
            try:
                path_len = nx.shortest_path_length(G, source, node)
                ctrl = max(ctrl, 1.0 / (1 + path_len))
            except nx.NetworkXNoPath:
                pass
        controllabilities.append(ctrl)
        
        # Observability: inverse of shortest path to any sink
        obs = 0
        for sink in list(sinks)[:10]:
            try:
                path_len = nx.shortest_path_length(G, node, sink)
                obs = max(obs, 1.0 / (1 + path_len))
            except nx.NetworkXNoPath:
                pass
        observabilities.append(obs)
    
    controllabilities = np.array(controllabilities)
    observabilities = np.array(observabilities)
    
    features['avg_controllability'] = np.mean(controllabilities)
    features['min_controllability'] = np.min(controllabilities)
    features['controllability_std'] = np.std(controllabilities)
    features['avg_observability'] = np.mean(observabilities)
    features['min_observability'] = np.min(observabilities)
    features['observability_std'] = np.std(observabilities)
    
    # Fraction of nodes with both low controllability and observability (trojan suspects)
    ctrl_threshold = np.percentile(controllabilities, 25)
    obs_threshold = np.percentile(observabilities, 25)
    low_both = np.sum((controllabilities <= ctrl_threshold) & (observabilities <= obs_threshold))
    features['low_ctrl_obs_fraction'] = low_both / len(sample_nodes)
    
    # Product of min controllability and observability
    features['ctrl_obs_product_min'] = np.min(controllabilities) * np.min(observabilities)
    
    return features


def compute_directed_graph_features(G: nx.DiGraph) -> Dict[str, float]:
    """Compute features specific to directed graphs."""
    features = {}
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        return {k: 0.0 for k in [
            'source_node_fraction', 'sink_node_fraction', 'isolated_node_fraction',
            'in_degree_mean', 'out_degree_mean', 'in_degree_std', 'out_degree_std',
            'in_out_degree_correlation', 'degree_asymmetry', 'reciprocity',
            'num_sccs', 'largest_scc_fraction', 'num_wccs', 'largest_wcc_fraction',
            'dag_longest_path', 'feedback_arc_fraction', 'transitivity'
        ]}
    
    # === In-degree / Out-degree analysis ===
    in_degrees = np.array([d for _, d in G.in_degree()])
    out_degrees = np.array([d for _, d in G.out_degree()])
    
    # Source nodes (no incoming edges) - potential trigger inputs
    source_nodes = np.sum(in_degrees == 0)
    features['source_node_fraction'] = source_nodes / num_nodes
    
    # Sink nodes (no outgoing edges) - potential payload outputs
    sink_nodes = np.sum(out_degrees == 0)
    features['sink_node_fraction'] = sink_nodes / num_nodes
    
    # Isolated nodes (no edges at all)
    isolated_nodes = np.sum((in_degrees == 0) & (out_degrees == 0))
    features['isolated_node_fraction'] = isolated_nodes / num_nodes
    
    # Degree statistics
    features['in_degree_mean'] = np.mean(in_degrees)
    features['out_degree_mean'] = np.mean(out_degrees)
    features['in_degree_std'] = np.std(in_degrees)
    features['out_degree_std'] = np.std(out_degrees)
    features['in_degree_max'] = np.max(in_degrees)
    features['out_degree_max'] = np.max(out_degrees)
    
    # Correlation between in and out degree (how hub-like are nodes?)
    if np.std(in_degrees) > 0 and np.std(out_degrees) > 0:
        features['in_out_degree_correlation'] = np.corrcoef(in_degrees, out_degrees)[0, 1]
    else:
        features['in_out_degree_correlation'] = 0
    
    # Degree asymmetry (difference between in and out for each node)
    degree_diff = np.abs(in_degrees - out_degrees)
    features['degree_asymmetry'] = np.mean(degree_diff) / max(np.mean(in_degrees + out_degrees), 1)
    
    # === Reciprocity (fraction of edges that are bidirectional) ===
    # High reciprocity might indicate feedback loops
    try:
        features['reciprocity'] = nx.reciprocity(G)
    except:
        features['reciprocity'] = 0
    
    # === Strongly Connected Components ===
    try:
        sccs = list(nx.strongly_connected_components(G))
        features['num_sccs'] = len(sccs)
        largest_scc = max(len(scc) for scc in sccs) if sccs else 0
        features['largest_scc_fraction'] = largest_scc / num_nodes
        # Non-trivial SCCs (size > 1) indicate cycles
        nontrivial_sccs = [scc for scc in sccs if len(scc) > 1]
        features['num_nontrivial_sccs'] = len(nontrivial_sccs)
        features['nontrivial_scc_node_fraction'] = sum(len(scc) for scc in nontrivial_sccs) / num_nodes
    except:
        features['num_sccs'] = 0
        features['largest_scc_fraction'] = 0
        features['num_nontrivial_sccs'] = 0
        features['nontrivial_scc_node_fraction'] = 0
    
    # === Weakly Connected Components ===
    try:
        wccs = list(nx.weakly_connected_components(G))
        features['num_wccs'] = len(wccs)
        largest_wcc = max(len(wcc) for wcc in wccs) if wccs else 0
        features['largest_wcc_fraction'] = largest_wcc / num_nodes
    except:
        features['num_wccs'] = 0
        features['largest_wcc_fraction'] = 0
    
    # === DAG-related features (after removing cycles) ===
    try:
        # Find feedback arc set (edges causing cycles)
        if nx.is_directed_acyclic_graph(G):
            features['feedback_arc_fraction'] = 0
            features['dag_longest_path'] = nx.dag_longest_path_length(G) / max(num_nodes, 1)
        else:
            # Approximate feedback arc set
            condensation = nx.condensation(G)
            features['dag_longest_path'] = nx.dag_longest_path_length(condensation) / max(num_nodes, 1)
            # Edges in SCCs are feedback edges
            feedback_edges = sum(len(scc) * (len(scc) - 1) for scc in sccs if len(scc) > 1)
            features['feedback_arc_fraction'] = feedback_edges / max(num_edges, 1)
    except:
        features['dag_longest_path'] = 0
        features['feedback_arc_fraction'] = 0
    
    # === Transitivity (clustering coefficient for directed graphs) ===
    try:
        features['transitivity'] = nx.transitivity(G)
    except:
        features['transitivity'] = 0
    
    # === Path-based features ===
    # Average shortest path in largest WCC
    try:
        if largest_wcc > 1:
            largest_wcc_nodes = max(wccs, key=len)
            subgraph = G.subgraph(largest_wcc_nodes)
            if nx.is_weakly_connected(subgraph):
                # Sample paths for large graphs
                if len(largest_wcc_nodes) > 100:
                    sample_nodes = list(largest_wcc_nodes)[:50]
                    paths = []
                    for src in sample_nodes[:10]:
                        for tgt in sample_nodes[10:20]:
                            if src != tgt:
                                try:
                                    paths.append(nx.shortest_path_length(subgraph, src, tgt))
                                except:
                                    pass
                    features['avg_path_length'] = np.mean(paths) if paths else 0
                else:
                    features['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            else:
                features['avg_path_length'] = 0
        else:
            features['avg_path_length'] = 0
    except:
        features['avg_path_length'] = 0
    
    # === Advanced directed graph features ===
    # Fanin/Fanout cone analysis
    cone_features = compute_fanin_fanout_cones(G)
    features.update(cone_features)
    
    # Topological features
    topo_features = compute_topological_features(G)
    features.update(topo_features)
    
    # Controllability/Observability estimates
    ctrl_obs_features = compute_controllability_observability(G)
    features.update(ctrl_obs_features)
    
    return features


def compute_graph_level_stats(graphs: List[dict]) -> Dict[str, np.ndarray]:
    """Compute graph-level statistics for a set of graphs."""
    stats_dict = defaultdict(list)
    
    for g in graphs:
        node_features = np.array(g['node_features'])
        num_nodes = g['num_nodes']
        num_edges = g['num_edges']
        # Support both old (num_cycles) and new (scc_size) data formats
        scc_size = g.get('scc_size', g.get('num_cycles', 0))
        edge_index = g['edge_index']
        
        # Build directed graph
        G = build_digraph_from_edge_index(edge_index, num_nodes)
        
        # Compute directed graph features
        directed_feats = compute_directed_graph_features(G)
        for feat_name, feat_val in directed_feats.items():
            stats_dict[feat_name].append(feat_val)
        
        # Basic graph stats
        stats_dict['num_nodes'].append(num_nodes)
        stats_dict['num_edges'].append(num_edges)
        stats_dict['scc_size'].append(scc_size)
        stats_dict['edge_density'].append(num_edges / max(num_nodes * (num_nodes - 1), 1))
        stats_dict['edges_per_node'].append(num_edges / max(num_nodes, 1))
        
        # Node feature statistics (per feature)
        for i, name in enumerate(FEATURE_NAMES):
            if i < node_features.shape[1]:
                feature_vals = node_features[:, i]
                stats_dict[f'{name}_mean'].append(np.mean(feature_vals))
                stats_dict[f'{name}_std'].append(np.std(feature_vals))
                stats_dict[f'{name}_max'].append(np.max(feature_vals))
                stats_dict[f'{name}_sum'].append(np.sum(feature_vals))
                # Fraction of nodes with this feature > 0.5
                stats_dict[f'{name}_frac'].append(np.mean(feature_vals > 0.5))
        
        # Gate type distribution
        gate_types = node_features[:, :14]  # First 14 features are one-hot gate types
        gate_counts = gate_types.sum(axis=0)
        total_gates = gate_counts.sum()
        
        if total_gates > 0:
            gate_fracs = gate_counts / total_gates
            for i, name in enumerate(FEATURE_NAMES[:14]):
                stats_dict[f'{name}_ratio'].append(gate_fracs[i])
        
        # Structural feature aggregations
        fan_in = node_features[:, 14]
        fan_out = node_features[:, 15]
        depth = node_features[:, 16]
        
        stats_dict['fan_in_variance'].append(np.var(fan_in))
        stats_dict['fan_out_variance'].append(np.var(fan_out))
        stats_dict['depth_variance'].append(np.var(depth))
        stats_dict['fan_ratio'].append(np.mean(fan_out) / max(np.mean(fan_in), 0.001))
        
        # Cycle-related features
        in_cycle = node_features[:, 18] if node_features.shape[1] > 18 else np.zeros(num_nodes)
        stats_dict['cycle_node_fraction'].append(np.mean(in_cycle))
        stats_dict['scc_size_ratio'].append(scc_size / max(num_nodes, 1))
        
        # Neighborhood features
        if node_features.shape[1] >= 26:
            inverter_density = node_features[:, 21]
            sequential_density = node_features[:, 22]
            rare_gate_density = node_features[:, 23]
            
            stats_dict['avg_inverter_density'].append(np.mean(inverter_density))
            stats_dict['avg_sequential_density'].append(np.mean(sequential_density))
            stats_dict['avg_rare_gate_density'].append(np.mean(rare_gate_density))
            stats_dict['max_rare_gate_density'].append(np.max(rare_gate_density))
            
            # Nodes with high rare gate density (potential trojan indicators)
            stats_dict['high_rare_gate_nodes'].append(np.sum(rare_gate_density > 0.1) / max(num_nodes, 1))
        
        # Graph-level features from the data
        if 'graph_features' in g:
            gf = np.array(g['graph_features'])
            for i, val in enumerate(gf):
                stats_dict[f'graph_feature_{i}'].append(val)
    
    return {k: np.array(v) for k, v in stats_dict.items()}


def statistical_comparison(trojaned_stats: Dict[str, np.ndarray], 
                          clean_stats: Dict[str, np.ndarray]) -> List[Tuple[str, float, float, float, float]]:
    """Compare statistics between trojaned and clean graphs."""
    results = []
    
    for feature_name in trojaned_stats.keys():
        if feature_name not in clean_stats:
            continue
        
        t_vals = trojaned_stats[feature_name]
        c_vals = clean_stats[feature_name]
        
        # Skip if all values are the same
        if np.std(t_vals) == 0 and np.std(c_vals) == 0:
            continue
        
        t_mean = np.mean(t_vals)
        c_mean = np.mean(c_vals)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(t_vals) + np.var(c_vals)) / 2)
        if pooled_std > 0:
            cohens_d = abs(t_mean - c_mean) / pooled_std
        else:
            cohens_d = 0
        
        # Statistical test (Mann-Whitney U for non-normal distributions)
        try:
            _, p_value = stats.mannwhitneyu(t_vals, c_vals, alternative='two-sided')
        except:
            p_value = 1.0
        
        results.append((feature_name, t_mean, c_mean, cohens_d, p_value))
    
    # Sort by effect size (Cohen's d)
    results.sort(key=lambda x: -x[3])
    return results


# =============================================================================
# Discriminative Feature Generation
# =============================================================================

def generate_discriminative_features(graph_data: dict) -> Dict[str, float]:
    """
    Generate discriminative features for a single graph based on 
    analysis of trojaned vs clean differences.
    """
    node_features = np.array(graph_data['node_features'])
    num_nodes = graph_data['num_nodes']
    num_edges = graph_data['num_edges']
    # Support both old (num_cycles) and new (scc_size) data formats
    scc_size = graph_data.get('scc_size', graph_data.get('num_cycles', 0))
    
    features = {}
    
    # === Size-normalized features (address size bias) ===
    features['log_num_nodes'] = np.log1p(num_nodes)
    features['log_num_edges'] = np.log1p(num_edges)
    features['log_scc_size'] = np.log1p(scc_size)
    features['scc_size_ratio'] = scc_size / max(num_nodes, 1)
    features['edges_per_node'] = num_edges / max(num_nodes, 1)
    
    # === Gate type ratios (size-invariant) ===
    gate_types = node_features[:, :14]
    gate_counts = gate_types.sum(axis=0)
    total_gates = gate_counts.sum()
    
    if total_gates > 0:
        gate_ratios = gate_counts / total_gates
        # Focus on suspicious gate types
        features['xor_xnor_ratio'] = (gate_ratios[7] + gate_ratios[8])  # XOR + XNOR
        features['rare_logic_ratio'] = (gate_ratios[7] + gate_ratios[8] + gate_ratios[12])  # XOR + XNOR + MUX
        features['inverter_ratio'] = (gate_ratios[4] + gate_ratios[6] + gate_ratios[8] + gate_ratios[9])  # NAND + NOR + XNOR + NOT
        features['sequential_ratio'] = gate_ratios[11]  # DFF
        features['wire_ratio'] = gate_ratios[2]  # Wire
        features['combinational_ratio'] = sum(gate_ratios[3:11]) - gate_ratios[11]  # All gates except DFF
    else:
        features['xor_xnor_ratio'] = 0
        features['rare_logic_ratio'] = 0
        features['inverter_ratio'] = 0
        features['sequential_ratio'] = 0
        features['wire_ratio'] = 0
        features['combinational_ratio'] = 0
    
    # === Structural feature statistics ===
    fan_in = node_features[:, 14]
    fan_out = node_features[:, 15]
    depth = node_features[:, 16]
    
    features['fan_in_mean'] = np.mean(fan_in)
    features['fan_out_mean'] = np.mean(fan_out)
    features['fan_in_std'] = np.std(fan_in)
    features['fan_out_std'] = np.std(fan_out)
    features['fan_asymmetry'] = abs(np.mean(fan_out) - np.mean(fan_in))
    features['depth_mean'] = np.mean(depth)
    features['depth_std'] = np.std(depth)
    features['depth_range'] = np.max(depth) - np.min(depth)
    
    # === Cycle-related features ===
    in_cycle = node_features[:, 18] if node_features.shape[1] > 18 else np.zeros(num_nodes)
    features['cycle_node_fraction'] = np.mean(in_cycle)
    features['scc_density'] = scc_size / max(num_nodes, 1)  # SCC size relative to subgraph
    
    # === Neighborhood features (if available) ===
    if node_features.shape[1] >= 26:
        avg_neighbor_fanin = node_features[:, 19]
        avg_neighbor_fanout = node_features[:, 20]
        inverter_density = node_features[:, 21]
        sequential_density = node_features[:, 22]
        rare_gate_density = node_features[:, 23]
        neighbor_count = node_features[:, 24]
        self_loop = node_features[:, 25]
        
        features['neighborhood_fanin_mean'] = np.mean(avg_neighbor_fanin)
        features['neighborhood_fanout_mean'] = np.mean(avg_neighbor_fanout)
        features['inverter_density_mean'] = np.mean(inverter_density)
        features['inverter_density_max'] = np.max(inverter_density)
        features['sequential_density_mean'] = np.mean(sequential_density)
        features['rare_gate_density_mean'] = np.mean(rare_gate_density)
        features['rare_gate_density_max'] = np.max(rare_gate_density)
        features['rare_gate_density_std'] = np.std(rare_gate_density)
        features['neighbor_count_mean'] = np.mean(neighbor_count)
        features['self_loop_fraction'] = np.mean(self_loop)
        
        # Suspicious pattern indicators
        features['high_rare_gate_nodes'] = np.sum(rare_gate_density > 0.1) / max(num_nodes, 1)
        features['low_connectivity_nodes'] = np.sum(neighbor_count < 0.05) / max(num_nodes, 1)
        
        # Heterogeneity measures (trojans often add anomalous nodes)
        features['rare_gate_heterogeneity'] = np.std(rare_gate_density) / max(np.mean(rare_gate_density), 0.001)
        features['inverter_heterogeneity'] = np.std(inverter_density) / max(np.mean(inverter_density), 0.001)
    
    # === Entropy-based features (complexity measures) ===
    # Gate type entropy
    if total_gates > 0:
        gate_probs = gate_ratios[gate_ratios > 0]
        features['gate_type_entropy'] = -np.sum(gate_probs * np.log2(gate_probs + 1e-10))
    else:
        features['gate_type_entropy'] = 0
    
    # Fan-in/fan-out distribution entropy
    fan_in_hist, _ = np.histogram(fan_in, bins=10, range=(0, 1))
    fan_in_probs = fan_in_hist / max(fan_in_hist.sum(), 1)
    fan_in_probs = fan_in_probs[fan_in_probs > 0]
    features['fan_in_entropy'] = -np.sum(fan_in_probs * np.log2(fan_in_probs + 1e-10))
    
    # === Directed graph features ===
    edge_index = graph_data['edge_index']
    G = build_digraph_from_edge_index(edge_index, num_nodes)
    directed_feats = compute_directed_graph_features(G)
    features.update(directed_feats)
    
    return features


def compute_all_discriminative_features(data: List[dict]) -> Tuple[np.ndarray, List[str]]:
    """Compute discriminative features for all graphs."""
    all_features = []
    feature_names = None
    
    for g in data:
        feats = generate_discriminative_features(g)
        if feature_names is None:
            feature_names = list(feats.keys())
        all_features.append([feats[name] for name in feature_names])
    
    return np.array(all_features), feature_names


# =============================================================================
# Feature Selection
# =============================================================================

def select_best_features(trojaned_data: List[dict], 
                         clean_data: List[dict],
                         top_k: int = 20) -> List[str]:
    """Select the most discriminative features."""
    
    # Compute discriminative features
    all_data = trojaned_data + clean_data
    labels = [1] * len(trojaned_data) + [0] * len(clean_data)
    
    features, feature_names = compute_all_discriminative_features(all_data)
    
    # Compute correlation with label
    correlations = []
    for i, name in enumerate(feature_names):
        feat_vals = features[:, i]
        # Skip constant features
        if np.std(feat_vals) == 0:
            correlations.append((name, 0, 1.0))
            continue
        
        # Point-biserial correlation
        try:
            corr, p_val = stats.pointbiserialr(labels, feat_vals)
            correlations.append((name, abs(corr), p_val))
        except:
            correlations.append((name, 0, 1.0))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: -x[1])
    
    print("\n" + "="*70)
    print("TOP DISCRIMINATIVE FEATURES (by correlation with label)")
    print("="*70)
    print(f"{'Feature':<40} {'|Corr|':>10} {'p-value':>12}")
    print("-"*70)
    
    selected = []
    for name, corr, p_val in correlations[:top_k]:
        print(f"{name:<40} {corr:>10.4f} {p_val:>12.2e}")
        if p_val < 0.1:  # Only include statistically significant features
            selected.append(name)
    
    return selected


# =============================================================================
# Output Enhanced Dataset
# =============================================================================

def save_enhanced_dataset(trojaned_data: List[dict], 
                          clean_data: List[dict],
                          output_path: str):
    """Save dataset with enhanced discriminative features."""
    
    all_data = trojaned_data + clean_data
    labels = [1] * len(trojaned_data) + [0] * len(clean_data)
    
    features, feature_names = compute_all_discriminative_features(all_data)
    
    # Create enhanced dataset
    enhanced_data = []
    for i, (g, label) in enumerate(zip(all_data, labels)):
        enhanced = {
            'design_id': g['design_id'],
            'original_node_features': g['node_features'],
            'edge_index': g['edge_index'],
            'edge_features': g['edge_features'],
            'discriminative_features': features[i].tolist(),
            'feature_names': feature_names,
            'label': label
        }
        enhanced_data.append(enhanced)
    
    with open(output_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"\nSaved enhanced dataset to {output_path}")
    print(f"  - {len(enhanced_data)} graphs")
    print(f"  - {len(feature_names)} discriminative features per graph")


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    """Run complete feature analysis."""
    
    print("="*70)
    print("CYCLE FEATURE ANALYZER")
    print("="*70)
    
    # Load data
    json_path = os.path.join(INPUT_DIR, "feedback_dataset.json")
    trojaned, clean = load_cycle_data(json_path)
    
    # Compute statistics
    print("\nComputing graph-level statistics...")
    trojaned_stats = compute_graph_level_stats(trojaned)
    clean_stats = compute_graph_level_stats(clean)
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (Top differences by effect size)")
    print("="*70)
    print(f"{'Feature':<35} {'Trojaned':>10} {'Clean':>10} {'Cohen d':>10} {'p-value':>10}")
    print("-"*70)
    
    comparison = statistical_comparison(trojaned_stats, clean_stats)
    for name, t_mean, c_mean, cohens_d, p_val in comparison[:30]:
        sig = "*" if p_val < 0.05 else ""
        print(f"{name:<35} {t_mean:>10.4f} {c_mean:>10.4f} {cohens_d:>10.4f} {p_val:>9.4f}{sig}")
    
    # Feature selection
    best_features = select_best_features(trojaned, clean, top_k=25)
    
    # Summary of key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Size comparison
    t_nodes = trojaned_stats['num_nodes']
    c_nodes = clean_stats['num_nodes']
    print(f"\n1. SIZE BIAS:")
    print(f"   Trojaned graphs: {np.mean(t_nodes):.0f} ± {np.std(t_nodes):.0f} nodes")
    print(f"   Clean graphs:    {np.mean(c_nodes):.0f} ± {np.std(c_nodes):.0f} nodes")
    print(f"   → Need size-invariant features!")
    
    # Gate type differences
    print(f"\n2. GATE TYPE DIFFERENCES:")
    for gate in ['gate_xor_ratio', 'gate_xnor_ratio', 'gate_dff_ratio', 'gate_mux_ratio']:
        if gate in trojaned_stats and gate in clean_stats:
            t_val = np.mean(trojaned_stats[gate])
            c_val = np.mean(clean_stats[gate])
            if abs(t_val - c_val) > 0.001:
                print(f"   {gate}: Trojaned={t_val:.4f}, Clean={c_val:.4f}")
    
    # Structural differences
    print(f"\n3. STRUCTURAL DIFFERENCES:")
    for feat in ['fan_in_variance', 'fan_out_variance', 'depth_variance', 'cycles_per_node']:
        if feat in trojaned_stats and feat in clean_stats:
            t_val = np.mean(trojaned_stats[feat])
            c_val = np.mean(clean_stats[feat])
            diff = abs(t_val - c_val) / max(c_val, 0.001)
            if diff > 0.1:
                print(f"   {feat}: Trojaned={t_val:.4f}, Clean={c_val:.4f} ({diff*100:.0f}% diff)")
    
    # Save enhanced dataset
    enhanced_path = os.path.join(OUTPUT_DIR, "enhanced_features_dataset.json")
    save_enhanced_dataset(trojaned, clean, enhanced_path)
    
    # Save feature importance report
    report_path = os.path.join(OUTPUT_DIR, "feature_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("CYCLE FEATURE ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("STATISTICAL COMPARISON (sorted by effect size)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Feature':<35} {'Trojaned':>10} {'Clean':>10} {'Cohen_d':>10} {'p-value':>10}\n")
        for name, t_mean, c_mean, cohens_d, p_val in comparison:
            f.write(f"{name:<35} {t_mean:>10.4f} {c_mean:>10.4f} {cohens_d:>10.4f} {p_val:>10.4f}\n")
        
        f.write("\n\nBEST DISCRIMINATIVE FEATURES:\n")
        for feat in best_features:
            f.write(f"  - {feat}\n")
    
    print(f"\nFeature analysis report saved to {report_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
