import numpy as np
import itertools
from scipy.stats import multivariate_normal
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd

# Define functions to calculate Mutual Information (MI) and Conditional Mutual Information (CMI)
def calculate_mi(x, y):
    joint = np.histogram2d(x, y, bins=30)[0]
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)
    pxy = joint / np.sum(joint)
    px = px / np.sum(px)
    py = py / np.sum(py)
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i][j] > 0:
                mi += pxy[i][j] * np.log(pxy[i][j] / (px[i] * py[j]))
    return mi

def calculate_cmi(x, y, z):
    xyz = np.vstack((x, y, *z)).T
    cov_xyz = np.cov(xyz, rowvar=False) + np.eye(xyz.shape[1]) * 1e-5  # Regularization term added for stability
    cov_xz = np.cov(xyz[:, [0] + list(range(2, 2 + len(z)))], rowvar=False) + np.eye(len([0] + list(range(2, 2 + len(z))))) * 1e-5
    cov_yz = np.cov(xyz[:, [1] + list(range(2, 2 + len(z)))], rowvar=False) + np.eye(len([1] + list(range(2, 2 + len(z))))) * 1e-5
    cov_z = np.cov(xyz[:, list(range(2, 2 + len(z)))], rowvar=False) + np.eye(len(list(range(2, 2 + len(z))))) * 1e-5

    mi_xz = 0.5 * np.log(np.linalg.det(cov_xz) / np.linalg.det(cov_z))
    mi_yz = 0.5 * np.log(np.linalg.det(cov_yz) / np.linalg.det(cov_z))
    mi_xyz = 0.5 * np.log(np.linalg.det(cov_xyz) / np.linalg.det(cov_z))

    return mi_xz + mi_yz - mi_xyz

# Helper function for multiprocessing
def process_edge(args):
    data, i, j, adj_genes, l, threshold, num_permutations, p_value_cutoff = args
    if l == 0:
        mi = calculate_mi(data[:, i], data[:, j])
        p_value = permutation_test(data[:, i], data[:, j], mi, num_permutations)
        if mi < threshold or p_value > p_value_cutoff:
            return (i, j, False)
    else:
        for subset in itertools.combinations(adj_genes, l):
            cmi = calculate_cmi(data[:, i], data[:, j], data[:, list(subset)])
            p_value = permutation_test(data[:, i], data[:, j], cmi, num_permutations, z=data[:, list(subset)])
            if cmi < threshold or p_value > p_value_cutoff:
                return (i, j, False)
    return (i, j, True)

# Permutation test for p-value
def permutation_test(x, y, original_stat, num_permutations, z=None):
    count = 0
    for _ in range(num_permutations):
        y_permuted = np.random.permutation(y)
        if z is None:
            perm_stat = calculate_mi(x, y_permuted)
        else:
            perm_stat = calculate_cmi(x, y_permuted, z)
        if perm_stat >= original_stat:
            count += 1
    p_value = count / num_permutations
    return p_value

# Path Consistency Algorithm (PCA) for inferring GRNs
def pca_cmi(data, threshold=0.03, order=None, num_permutations=100, p_value_cutoff=0.05):
    num_genes = data.shape[1]
    graph = np.ones((num_genes, num_genes)) - np.eye(num_genes)  # Complete graph with no self-loops
    
    # Step 0: Initialization
    l = -1
    pool = mp.Pool(mp.cpu_count())
    while True:
        l += 1
        if order is not None and l > order:
            break
        changes = False
        tasks = (
            (data, i, j, adj_genes, l, threshold, num_permutations, p_value_cutoff)
            for (i, j) in itertools.combinations(range(num_genes), 2)
            if graph[i, j] != 0
            for adj_genes in [[k for k in range(num_genes) if k != i and k != j and graph[i, k] and graph[j, k]]]
            if len(adj_genes) >= l
        )
        
        results = []
        for result in tqdm(pool.imap(process_edge, tasks, chunksize=10), total=num_genes * (num_genes - 1) // 2):
            results.append(result)
        
        for i, j, keep in results:
            if not keep:
                graph[i, j] = graph[j, i] = 0
                changes = True
        
        if not changes:
            break
    pool.close()
    pool.join()
    
    return graph

# Output GRN in table format
def output_grn(graph):
    num_genes = graph.shape[0]
    edges = []
    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            if graph[i, j] == 1:
                edges.append([f"Gene{i + 1}", f"Gene{j + 1}"])
    grn_df = pd.DataFrame(edges, columns=["Gene1", "Gene2"])
    return grn_df

# Example Usage
data = np.random.rand(100, 10)  # 100 samples of 10 genes
grn_graph = pca_cmi(data, order=2, num_permutations=100)
gr_df = output_grn(grn_graph)
print("Inferred Gene Regulatory Network in Table Format:")
print(grn_df)
