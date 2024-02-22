import itertools
import math
import pickle
import random
import statistics
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import sklearn.neighbors as sk
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

constants = {
    'cost_transmission': 1.,
    'recharge_rate': 0.,
    'initial_charge': 0.
}


def make_3dgraph(x: list, y: list, z: list, x_label: str, y_label: str, z_label: str, filename: str, flatten_y=False):
    if flatten_y:
        make_2dgraph(x, z, x_label=x_label, y_label=z_label, filename=filename)
        return
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter3D(x, y, z, c=y, cmap='viridis_r')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.savefig(filename)
    plt.close()


def make_2dgraph(x: list, y: list, x_label: str, y_label: str, filename: str):
    ax = plt.figure().add_subplot()
    ax.set_ylim([0, 1])
    ax.scatter(x, y, c=y, cmap='viridis_r', clim=[0, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(filename)
    plt.close()


def make_2dmultigraph(x: list, y: list, z: list, x_label: str, y_label: str, z_label: str, filename: str,
                      flatten_y=False):
    if flatten_y:
        make_2dgraph(x, z, x_label=x_label, y_label=z_label, filename=filename)
        return
    ax = plt.figure().add_subplot()
    for index in y:
        ax.scatter(x, z, c=y, cmap='viridis', alpha=0.4, label=f'{index}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(z_label)
    plt.savefig(filename)
    plt.close()


def make_histogram(histogram, filename: str):
    hist = plt.hist(histogram, bins='auto', density=True)
    plt.savefig(filename)
    plt.close()


def create_knn_graph(n: int, k: int):
    points = np.random.default_rng().uniform(size=(n, 3))
    knn = sk.kneighbors_graph(points, p=2, mode='distance', n_neighbors=k)
    knngraph = nx.from_scipy_sparse_array(knn, create_using=nx.DiGraph)
    return knngraph


def count_reachable_nodes(graph: nx.DiGraph):
    return len(max(nx.strongly_connected_components(graph), key=len))


def assign_battery_charge(graph: nx.DiGraph, charge=constants['initial_charge']):
    for node in graph.nodes:
        graph.nodes[node]['charge'] = charge


def compute_node_TX_costs(graph, node_label):
    graph.nodes[node_label]['tx_cost'] = (constants['cost_transmission'] *
                                          (sorted(graph.out_edges(node_label, data=True), key=lambda x: x[2]['weight'])[
                                              -1][2]['weight']) ** 2)


def update_battery(graph):
    for i in graph.nodes:
        # DRAINING
        # Transmission costs
        compute_node_TX_costs(graph, i)
        graph.nodes[i]['charge'] -= graph.nodes[i]['tx_cost']


def get_mean_connectivity(n: int, k: int, trials: int):
    results = []
    for i in range(trials):
        graph = create_knn_graph(n, k)
        results.append(count_reachable_nodes(graph))
    return statistics.mean(results)


def run_connectivity_experiment(ns: list, ks: list, trials: int = 10000):
    results = []
    for n in ns:
        for k in ks:
            print(n, k)
            if n > k:
                results.append([n, k, get_mean_connectivity(n, k, trials) / float(n)])
    return results


def scenario_connectivity(note: str = '',
                          graph_sizes=None,
                          k_list=None,
                          trials=300):
    if graph_sizes is None:
        graph_sizes = list(range(10, 1000, 10))
    if k_list is None:
        k_list = list(range(1, 13, 1))
    data = run_connectivity_experiment(graph_sizes, k_list, trials=trials)
    with open(f'conn{note}.txt', 'w') as outfile:
        outfile.write('N -- n. of vertices, k -- which k-NN, LSCC rel. size\n\n')
        for i in data:
            outfile.write(', '.join(str(j) for j in i))
            outfile.write('\n')
    x, y, z = zip(*data)
    make_2dmultigraph(x, y, z, x_label='number of vertices', y_label='k', z_label='relative size of LSCC.',
                      filename=f'fig_conn_{note}.pdf')


def simulate_battery_drain(n=100, k=2, timehorizon=100):
    graph = create_knn_graph(n, k)
    assign_battery_charge(graph)
    for i in range(timehorizon):
        update_battery(graph)
    charges = [x[1]['charge'] for x in graph.nodes(data=True)]
    tx_costs = [x[1]['tx_cost'] for x in graph.nodes(data=True)]
    # nz_charges = [x for x in charges if x > 0]
    return charges, tx_costs


def simulate_job_allocation(n, k, percentile, beta=1):
    if n == 0:
        raise ValueError
    distribution = scipy.stats.gengamma(a=k, c=3. / 2, loc=0, scale=(0.685 / (n ** (22. / 30))))
    graph = create_knn_graph(n, k)
    assign_battery_charge(graph)
    update_battery(graph)
    # maximal_tx = max(all_tx_costs)
    maximal_tx = distribution.ppf(percentile)
    assign_battery_charge(graph, maximal_tx)
    old_size_of_LSCC = len(max(nx.strongly_connected_components(graph), key=len))

    overloaded_sats = [n for n in graph.nodes(data=True) if n[1]['tx_cost'] > n[1]['charge']]
    for i in overloaded_sats:
        out_edges = sorted(graph.out_edges(i[0], data=True), key=lambda x: x[2]['weight'], reverse=True)
        for edge in out_edges:
            graph.remove_edge(edge[0], edge[1])
            remaining_out_edges = sorted(graph.out_edges(i[0], data=True), key=lambda x: x[2]['weight'], reverse=True)
            if not len(remaining_out_edges) == 0:
                compute_node_TX_costs(graph, i[0])
            if i[1]['tx_cost'] <= i[1]['charge']:
                break
        out_edges = sorted(graph.out_edges(i[0], data=True), key=lambda x: x[2]['weight'], reverse=True)
        pass
    size_of_LSCC = len(max(nx.strongly_connected_components(graph), key=len))
    wtf = [n for n in graph.nodes(data=True) if n[1]['tx_cost'] > n[1]['charge']]
    graph.remove_nodes_from([n for n in graph if n not in max(nx.strongly_connected_components(graph), key=len)])
    all_tx_costs = [[x[0], x[1]['tx_cost']] for x in graph.nodes(data=True)]
    job_list = list(itertools.combinations([x[0] for x in all_tx_costs], 2))
    comp_cost = (n * beta * distribution.ppf(percentile)) / float(math.comb(n, 2))
    # comp_cost = (sum([beta*distribution.ppf(percentile) for i in range(1, n)]) / len(job_list))

    for satellite in graph.nodes(data=True):
        satellite[1]['jobs'] = []
        previously_computed = 0
        preceding_sats = [x[0] for x in graph.nodes(data=True) if x[0] < satellite[0]]
        for i in range(len(preceding_sats)):
            previously_computed = previously_computed + int((satellite[1]['charge'] - all_tx_costs[i][1]) / comp_cost)
        my_capacity = int((satellite[1]['charge'] - satellite[1]['tx_cost']) / comp_cost)
        assert my_capacity >= 0, 'computation capacity cannot be negative!!!'
        if previously_computed > len(job_list):
            continue
        my_choices = list(range(my_capacity))
        for choice in my_choices:
            if previously_computed + choice < len(job_list):
                satellite[1]['jobs'].append(job_list[previously_computed + choice])

    selected_jobs = []
    for satellite in graph.nodes(data=True):
        for job in satellite[1]['jobs']:
            selected_jobs.append(job)

    assert len(selected_jobs) > 0, f'No jobs assigned!'
    assert (len(set(selected_jobs)) / len(selected_jobs)) == 1, f'At least 1 job was assigned twice!'

    return len(set(selected_jobs)) / len(job_list), len(selected_jobs) / math.comb(n, 2), 1 - (
            len(set(selected_jobs)) / len(selected_jobs)), size_of_LSCC / old_size_of_LSCC


def scenario_battery_drain(note: str = '',
                           graph_sizes=None,
                           k_list=None,
                           timehorizon: int = 1000,
                           trials: int = 300):
    results = []
    if graph_sizes is None:
        graph_sizes = list(range(10, 1000, 10))
    if k_list is None:
        k_list = list(range(1, 13, 1))
    for n in graph_sizes:
        for k in k_list:
            print(n, k)
            if n > k:
                for i in range(trials):
                    charges, tx_costs = simulate_battery_drain(n, k, timehorizon)
                    # print( f'Average charge: {statistics.mean(charges)}; Average tx cost: {statistics.mean(
                    # tx_costs)}, total tx cost: {sum(tx_costs)}')
                    nz_charges = [x for x in charges if x > 0]
                    results.append(
                        [n, k, statistics.mean(charges), statistics.mean(nz_charges) if len(nz_charges) > 0 else 0,
                         len(charges) - len(nz_charges), statistics.mean(tx_costs)])
    with open(f'batt{note}.txt', 'w') as outfile:
        outfile.write(
            'N -- n. of vertices, k -- which k-NN, mean batt. charge, mean non-zero charge, n. of non-zero charges, '
            'mean TX cost\n\n')
        for i in results:
            outfile.write(', '.join(str(j) for j in i))
            outfile.write('\n')
    x, y, z_mean_charges, z_mean_pos_charges, z_count_zero_charges, z_mean_tx_costs = zip(*results)
    # print(list(z_mean_tx_costs))
    make_3dgraph(list(x), list(y), list(z_mean_charges), x_label='number of vertices', y_label='k',
                 z_label='mean charge (incl. zeros)', filename=f'fig_meanchg_{note}.pdf')
    make_3dgraph(list(x), list(y), list(z_mean_tx_costs), x_label='number of vertices', y_label='k',
                 z_label='mean transmission costs',
                 filename=f'fig_meantxc_{note}.pdf')


def scenario_tx_costs(note: str = '',
                      graph_sizes=None,
                      k_list=None,
                      generate_dists=False,
                      trials: int = 300):
    results = []
    if graph_sizes is None:
        graph_sizes = list(range(10, 1000, 10))
    if k_list is None:
        k_list = list(range(1, 13, 1))

    all_tx_costs = []
    scaling_factors = []

    for n in graph_sizes:
        for k in k_list:
            print(n, k)
            if n > k:
                for i in range(trials):
                    charges, tx_costs = simulate_battery_drain(n, k, 1)
                    results.append(
                        [n, k, statistics.mean(tx_costs), max(tx_costs), sum(tx_costs)])
                    all_tx_costs.extend(tx_costs)
                guess_a = k
                guess_c = 3. / 2
                guess_scale = (0.685 / (n ** (22. / 30)))
                print(f'parameter s: {guess_scale}')
                # expected_distr = scipy.stats.gengamma(guess_a, guess_c, scale=guess_scale)
                # a1, b1, loc1, scale1 = scipy.stats.gengamma.fit(all_tx_costs, guess_a, guess_c, fix_a=guess_a,
                #                                                 fix_c=guess_c, floc=0, scale=guess_scale, method='MM')
                # print(f'n={n}, k={k}, a={a1}, c={b1}, loc={loc1}, scale={scale1}')
                print(statistics.mean(all_tx_costs))
                make_histogram(all_tx_costs, f'hist_{n}_{k}_{note}.pdf')
                if generate_dists:
                    distr = scipy.stats.rv_histogram(np.histogram(all_tx_costs, bins=n))
                    with open(f'distr-{n}-{k}.pickle', 'wb') as f:
                        pickle.dump(distr, f)
                    with open(f'data-{n}-{k}.pickle', 'wb') as f:
                        pickle.dump(all_tx_costs, f)
                # scaling_factors.append(scale1 * (n ** (22. / 30)))
                all_tx_costs.clear()
    # print(scaling_factors)
    # try:
    #     make_2dgraph(list(graph_sizes), scaling_factors, x_label='n', y_label='scaling factor',
    #              filename=f'scales_{note}.pdf')
    # except:
    #     print(f'scales_{note}.pdf FAILED')
    # log_GS = [math.log(i) for i in graph_sizes]
    # log_scales = [math.log(i) for i in scaling_factors]
    # try:
    #     make_2dgraph(list(log_GS), log_scales, x_label='log n', y_label='log SF',
    #              filename=f'logscales_{note}.pdf')
    # except:
    #     print(f'logscales_{note}.pdf FAILED')
    # print(f'Mean scaling factor: {statistics.mean(scaling_factors)}')
    # regression = scipy.stats.linregress(log_GS, log_scales)
    # print(
    #     f'slope:{regression.slope}, intercept:{regression.intercept}, serr:{regression.stderr}, ierr:{regression.intercept_stderr}')
    with open(f'tx{note}.txt', 'w') as outfile:
        outfile.write('N -- n. of vertices, k -- which k-NN, mean TX cost, max TX cost, total TX cost\n\n')
        for i in results:
            outfile.write(', '.join(str(j) for j in i))
            outfile.write('\n')
    x, y, z_mean_tx_costs, z_max_tx_costs, z_total_tx_costs = zip(*results)
    make_3dgraph(list(x), list(y), list(z_mean_tx_costs), x_label='number of vertices', y_label='k',
                 z_label='mean tx cost', filename=f'fig_meantx_{note}.pdf')
    make_3dgraph(list(x), list(y), list(z_total_tx_costs), x_label='number of vertices', y_label='k',
                 z_label='total tx cost', filename=f'fig_totaltx_{note}.pdf')
    make_3dgraph(list(x), list(y), list(z_max_tx_costs), x_label='number of vertices', y_label='k',
                 z_label='max tx cost',
                 filename=f'fig_maxtx_{note}.pdf')


def scenario_job_allocation(note: str = '',
                            graph_sizes=None,
                            k_list=None,
                            trials: int = 300,
                            percentile=99.,
                            beta=1,
                            empirical_dists=True):
    results = []
    if graph_sizes is None:
        graph_sizes = list(range(10, 1000, 10))
    if k_list is None:
        k_list = list(range(1, 13, 1))

    for n in graph_sizes:
        for k in k_list:
            print(n, k, percentile, beta)
            if n > k:
                if empirical_dists:
                    for i in range(trials):
                        try:
                            with open(f'distr-{n}-{k}.pickle', 'rb') as f:
                                with open(f'distr-{n}-{k - 1}.pickle', 'rb') as f2:
                                    cover_frac, tot_job_frac, red_jobs_frac, LSCC_size = simulate_job_allocation(n, k,
                                                                                                                 percentile=percentile,
                                                                                                                 beta=beta)
                        except:
                            cover_frac, tot_job_frac, red_jobs_frac, LSCC_size = simulate_job_allocation(n, k,
                                                                                                         percentile=percentile,
                                                                                                         beta=beta)
                        results.append([n, k, cover_frac, tot_job_frac, red_jobs_frac, LSCC_size])
                else:
                    output = Parallel(n_jobs=7)(
                        delayed(simulate_job_allocation)(n, k, percentile, beta) for i in range(trials))
                    for i in output:
                        results.append([n, k, i[0], i[1], i[2], i[3]])
    with open(f'txtdata_{"".join(str(x) for x in k_list)}-{int(percentile)}-{beta}-{note}.txt', 'w') as outfile:
        outfile.write(
            'N -- n. of vertices, k -- which k-NN, coverage fraction wrt LSCC, coverage fraction wrt swarm, fraction of redundant jobs, LSCC fraction\n\n')
        for i in results:
            outfile.write(', '.join(str(j) for j in i))
            outfile.write('\n')
    x, y, z_coverage, z_tot_jobs, z_redundant, z_LSCC_size = zip(*results)
    flatten_k = False
    if len(k_list) == 1:
        flatten_k = True
    make_3dgraph(list(x), list(y), list(z_coverage), x_label='number of vertices', y_label='k',
                 z_label='jobs computed / all possible jobs in the LSCC',
                 filename=f'fig_coverLSCC_{"".join(str(x) for x in k_list)}-{percentile:.3f}-{beta:.3f}-{note}.pdf',
                 flatten_y=flatten_k)
    make_3dgraph(list(x), list(y), list(z_tot_jobs), x_label='number of vertices', y_label='k',
                 z_label='jobs computed / all possible jobs in the swarm',
                 filename=f'fig_coverALL_{"".join(str(x) for x in k_list)}-{percentile:.3f}-{beta:.3f}-{note}.pdf',
                 flatten_y=flatten_k)
    make_3dgraph(list(x), list(y), list(z_LSCC_size), x_label='number of vertices', y_label='k',
                 z_label='size of LSCC post-pruning / size of LSCC pre-pruning',
                 filename=f'fig_fracLSCC_{"".join(str(x) for x in k_list)}-{percentile:.3f}-{beta:.3f}-{note}.pdf',
                 flatten_y=flatten_k)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for k in range(4, 7):
        for percentile in np.arange(0.91, 1.0, 0.01):
            for beta in np.arange(0.1, 1.01, 0.1):
                scenario_job_allocation(graph_sizes=range(100, 1001, 100), k_list=[k], trials=100, note=f'Oct5',
                                        empirical_dists=False, percentile=percentile, beta=beta)
