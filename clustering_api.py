import pandas as pd
import numpy as np

from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


phone_list = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh',
       'er', 'ey', 'f', 'g', 'h', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng',
       'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw',
       'v', 'w', 'y', 'z', 'zh']

vowel_set = {'ih', 'uh', 'ay', 'oy', 'ow', 'ey', 'er', 'aa', 'ah', 'uw', 'ao', 'iy', 'ax', 'eh', 'aw', 'ae'}

kl_divergence = lambda point1, point2: ((point1 - point2)*(np.log(point1)-np.log(point2))).sum()

l2_distance = lambda point1, point2: ((point1 - point2)**2).sum()


class Clusterer(object):
    """Class to cluster the PPG features"""
    def __init__(self, n_cluster = 12, distance_func = "kl_divergence", minimum_cluster_size=3, minimum_sampe_for_cluster=20, k=0.1, discard_distance_threshold=0.33, minimum_distance=0.45, random_state=42):
        """
        :param n_cluster: The cluster amount in K-Means algorithm
        :param distance_func: The name of distance metric function, should be either "kl_divergence" or "l2_distance"
        :param minimum_cluster_size: The clusters with samples amount lower than minimum_cluster_size will be dicarded
        :param minimum_sampe_for_cluster: The minimum samples amount to conduct a clustering
        :param k: The multiplier used to filter impure clusters
        :param discard_distance_threshold: The distance threshold to decide whether a cluster is impure and to be discard
        :param minimum_distance: The minimum distance to decide whether to merge two similar clusters
        :param random_state: The random_state used in K-Means algorithm
        """
        super(Clusterer, self).__init__()
        self.n_cluster = n_cluster
        if distance_func == 'kl_divergence':
            self.distance_func = kl_divergence
        elif distance_func == 'l2_distance':
            self.distance_func = l2_distance
        else:
            raise ValueError("distance_func should be either \'kl_divergence\' or \'l2_distance\'")
        self.minimum_cluster_size = minimum_cluster_size
        self.minimum_sampe_for_cluster = minimum_sampe_for_cluster
        self.k = k
        self.discard_distance_threshold = discard_distance_threshold
        self.minimum_distance = minimum_distance
        self.random_state = random_state

    def cluster(self, X: np.ndarray, utterance_id_list: list) -> list:
        assert X.shape[1] == len(phone_list)
        df = pd.DataFrame(X, columns=phone_list)
        df['utterance_id'] = utterance_id_list       
        # filter the consonants
        df["top_1_p"] = df[phone_list].idxmax(axis=1)
        df = df[df["top_1_p"].isin(vowel_set)]
        if len(df) < self.minimum_sampe_for_cluster:
            return []
        X = df[phone_list].values

        # K-Means clustering
        metric = distance_metric(type_metric.USER_DEFINED, func=self.distance_func)
        initial_centers = kmeans_plusplus_initializer(X, self.n_cluster, random_state=self.random_state).initialize()
        kmeans_instance = kmeans(X, initial_centers, metric=metric)
        kmeans_instance.process()
        cluster_labels = np.zeros(len(X))
        for i, c in enumerate(kmeans_instance.get_clusters()):
            cluster_labels[c] = i
        df['cluster'] = cluster_labels.astype(int).astype(str)

        # discard impure clusters
        center_df = df.groupby('cluster')[phone_list].mean().T
        distance_list = []
        distance_std_list = []
        for i in range(center_df.shape[1]):
            centroid = center_df[center_df.columns[i]].values
            samples = df[df['cluster']==center_df.columns[i]][phone_list].values
            distances = []
            for j in range(len(samples)):
                distances.append(self.distance_func(centroid, samples[j, :]))
            distance_list.append(np.mean(distances))
            distance_std_list.append(np.std(distances))
        distance_sr = pd.Series(distance_list, index=center_df.columns)
        tmp_ = distance_sr[distance_sr>0]
        median_, std_ = tmp_.median(), tmp_.std()

        cluster_size = df.groupby('cluster').size()

        to_filter_cluster= distance_sr[(distance_sr>self.discard_distance_threshold)&(distance_sr>median_ + self.k*std_)].index.to_list() + cluster_size[cluster_size<self.minimum_cluster_size].index.to_list()
        discarded_center_df = center_df.drop(to_filter_cluster, axis=1)
        amt_cluster_after_discard = self.n_cluster - len(to_filter_cluster)
        if amt_cluster_after_discard == 0:
            return []

        # distance inter clusters
        xy_list = [discarded_center_df.values[:, i] for i in range(discarded_center_df.shape[1])]
        dm = np.asarray([[self.distance_func(p1, p2) for p2 in xy_list] for p1 in xy_list])
        distance_inter_clusters = pd.DataFrame(dm, index=discarded_center_df.columns, columns=discarded_center_df.columns)

        # merge similar clusters
        close_list = []
        for i in range(distance_inter_clusters.shape[0]):
            for j in range(distance_inter_clusters.shape[1]):
                if i < j and distance_inter_clusters.values[i,j] <  self.minimum_distance:
                    close_list.append((distance_inter_clusters.index[i], distance_inter_clusters.columns[j], distance_inter_clusters.values[i,j]))

        to_merge_list = []
        for a, b, _ in close_list:
            if not to_merge_list:
                to_merge_list.append({a, b})
                continue
            flag = -1
            ll = []
            for i, s in enumerate(to_merge_list):
                if a in s or b in s:
                    ll.append(i)
            if len(ll) > 1:
                ss = set()
                for i in ll:
                    ss = ss.union(to_merge_list[i])
                to_merge_list = [x for i, x in enumerate(to_merge_list) if i not in ll]
                to_merge_list.append(ss)
            elif len(ll) == 1:
                to_merge_list[ll[0]] = to_merge_list[ll[0]].union({a, b})

            else:
                to_merge_list.append({a, b})

        # map original clusters to discarded or merged clusters
        map_dict = dict()
        for s in to_merge_list:
            l = list(s)
            new_c = '-'.join(l)
            for c in l:
                map_dict[c] = new_c
        for c in to_filter_cluster:
            map_dict[c] = f'{c}-discarded'
        for c in df['cluster'].unique():
            if c not in map_dict:
                map_dict[c] = c
        df['cluster_id'] = df['cluster'].map(map_dict)

        refined_center_df = df.groupby('cluster_id')[phone_list].mean().T
        refined_center_df = refined_center_df[[x for x in refined_center_df.columns if not x.endswith('discarded')]]
        # refined_center_df = refined_center_df.T.reset_index()

        amt_cluster_after_merge = refined_center_df.shape[1]

        # output
        output_list = []
        for c in refined_center_df.columns:
            within_c_df = df[df['cluster_id']==c].copy()
            distances = []
            for j in range(len(within_c_df)):
                distances.append(self.distance_func(centroid, within_c_df[phone_list].values[j, :]))
            within_c_df['dist'] = distances

            output_list.append({
                'cluster_id': c, 
                'cluster_size': len(within_c_df),
                'utterance_id_list': ';'.join(within_c_df.sort_values('dist')['utterance_id'].head(10).to_list()),  # only top 10
                'centroid': refined_center_df[c].to_list(),
                'dist_mean': np.mean(distances),
                'dist_std': np.std(distances)
                })

        return output_list


if __name__ == '__main__': 
    df = pd.read_pickle('test_df')
    
    clr = Clusterer(n_cluster=12, distance_func="kl_divergence", minimum_cluster_size=3, minimum_sampe_for_cluster=20, k=0.1, discard_distance_threshold=0.33, minimum_distance=0.4)
    result = clr.cluster(df[phone_list].values, df[2].to_list())
    print(result)

    