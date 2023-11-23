import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
import seaborn as sns

def define_variable_clustering(df):
    summary = df.groupby('customer_unique_id').agg({
        'customer_state': 'first',
        'order_purchase_in_months': 'min',
        'order_id': 'nunique',
        'payment_value': 'median',
        'review_score': 'median',
        'payment_installments': 'median',
        'price': 'median',
        'freight_value': 'median'
    }).reset_index()

    # Renommage
    summary.rename(columns={
        'review_score': 'Score_median',
        'payment_installments': 'Payment_installments_median',
        'price': 'Median_price_product',
        'freight_value': 'freight_value_median'
    }, inplace=True)

    min_purchase_indices = df.groupby('customer_unique_id')['order_purchase_in_months'].idxmin()
    last_review_scores = df.loc[min_purchase_indices, ['customer_unique_id', 'review_score']]
    last_review_scores.rename(columns={'review_score': 'last_order_review_score'}, inplace=True)

    summary = summary.merge(last_review_scores, on='customer_unique_id', how='left')

    # Calcul de la moyenne du nombre de produits par commande et par customer_id
    avg_products_per_order = df.groupby(['order_id', 'customer_unique_id'])['product_id'].count().reset_index()
    avg_products_per_order = avg_products_per_order.groupby('customer_unique_id')['product_id'].mean().reset_index()
    avg_products_per_order.rename(columns={'product_id': 'avg_products_per_order'}, inplace=True)

    summary = summary.merge(avg_products_per_order, on='customer_unique_id', how='left')

    summary.rename(columns={
        'order_purchase_in_months': 'last_order_purchase_in_months',
        'order_id': 'number_of_orders',
        'payment_value': 'Panier_median'
    }, inplace=True)

    return summary


def plot_cluster_stats(df, cluster_var):
    cluster_stats = df.groupby(cluster_var).agg({
        'Panier_median': ['sum','median'],
        'customer_unique_id': ['count'],
        'last_order_purchase_in_months': ['median'],
        'number_of_orders': ['min', 'max'],
        'Score_median': ['median'],
        'Payment_installments_median': ['median'],
        'Median_price_product': ['median'],
        'freight_value_median': ['median'],
        'avg_products_per_order': ['median']
    })

    total_CA = df['Panier_median'].sum()
    total_clients = df['customer_unique_id'].nunique()
    cluster_stats['percentage_CA'] = (
        cluster_stats['Panier_median']['sum'] / total_CA) * 100

    cluster_stats['percentage_clients'] = (
        cluster_stats['customer_unique_id']['count'] / total_clients) * 100

    pourcentage_CA = cluster_stats['percentage_CA']
    pourcentage_clients = cluster_stats['percentage_clients']
    last_order_purchase_median = cluster_stats[
        'last_order_purchase_in_months']['median']
    score_median_median = cluster_stats['Score_median']['median']
    number_of_orders_min = cluster_stats['number_of_orders']['min']
    number_of_orders_max = cluster_stats['number_of_orders']['max']
    Panier_median_median = cluster_stats['Panier_median']['median']
    Payment_installments_median_median = cluster_stats['Payment_installments_median']['median']
    Median_price_product_median = cluster_stats['Median_price_product']['median']
    freight_value_median_median = cluster_stats['freight_value_median']['median']
    avg_products_per_order_median = cluster_stats['avg_products_per_order']['median']

    labels = pourcentage_CA.index
    colors = sns.color_palette("Set3", n_colors=len(labels))
    colors2 = sns.color_palette(n_colors=len(labels))
    fig, axs = plt.subplots(4, 3, figsize=(30, 20))

    axs[0,0].pie(pourcentage_CA,
              labels=labels,
              autopct='%1.1f%%',
              startangle=90,
              colors=colors)

    axs[0,0].set_title('Répartition par Cluster (Percentage CA)')

    axs[0,1].pie(pourcentage_clients,
              labels=labels,
              autopct='%1.1f%%',
              startangle=90,
              colors=colors)

    axs[0,1].set_title('Répartition par Cluster (Percentage Clients)')

    axs[1,0].bar(labels, last_order_purchase_median, color=colors)
    axs[1,0].set_title('Last Order Purchase (Mean) par Cluster')

    axs[1,1].bar(labels,
              score_median_median,
              color=colors)

    axs[1,1].set_title('Score Median (Median) par Cluster')
    axs[1,1].set_xlabel('Cluster')
    axs[1,1].set_ylabel('Score Median')

    axs[1,2].bar(labels,
              number_of_orders_max,
              label='Number of Orders (Max)',
              color=colors)

    axs[1,2].bar(labels,
              number_of_orders_min,
              label='Number of Orders (Min)',
              color=colors2)

    axs[1,2].set_xlabel('Cluster')
    axs[1,2].set_ylabel('Number of Orders')
    axs[1,2].set_title('Number of Orders (Min and Max) par Cluster')

    axs[2,0].bar(labels,
              Payment_installments_median_median,
              label='Payment_installments_median par Cluster',
              color=colors)
    axs[2,0].set_title('Répartition par Cluster (Payment_installments_median)')
    axs[2,0].set_xlabel('Cluster')
    axs[2,0].set_ylabel('Payment_installments_median')

    # Répartition par Cluster (Median_price_product)
    axs[2,1].bar(labels,
                Median_price_product_median,
                label='Payment_installments_median par Cluster',
                color=colors)
    axs[2,1].set_title('Répartition par Cluster (Median_price_product)')
    axs[2,1].set_xlabel('Cluster')
    axs[2,1].set_ylabel('Median_price_product')

    # Répartition par Cluster (freight_value_median)
    axs[2,2].bar(labels,
                freight_value_median_median,
                label='freight_value_median par Cluster',
                color=colors)
    axs[2,2].set_title('Répartition par Cluster (freight_value_median)')

    # Répartition par Cluster (panier_median)
    axs[3,0].bar(labels,
                Panier_median_median,
                label='Panier_median',
                color=colors)
    axs[3,0].set_title('Répartition par Cluster (Panier_median)')

    # Répartition par Cluster (avg_products_per_order)
    axs[0,2].bar(labels,
                avg_products_per_order_median,
                label='freight_value_median par Cluster',
                color=colors)
    axs[0,2].set_title('Répartition par Cluster (avg_products_per_order)')

    plt.tight_layout()
    plt.show()
    
    
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

def clustering_choice(model, range_n_clusters):
    # Score de silhouette
    silhouette_score_avg = []
    # Liste du nombre de clusters à tester
    palette = sns.color_palette("Set3", max(range_n_clusters))
    colors_2 = palette.as_hex()

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig = plt.figure(1, figsize=(18, 7))
        ax1 = fig.add_subplot(121)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example
        # all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(model) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters,
                           init='k-means++',
                           random_state=42)
        # Train
        cluster_labels = clusterer.fit_predict(model)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation
        # of the formed clusters
        silhouette_score_avg.append(silhouette_score(model, cluster_labels))

        # Compute the silhouette scores for each sample
        sample_silh_values = silhouette_samples(model, cluster_labels)

        tnse = TSNE()
        x_tsne = tnse.fit_transform(model)
        x_tsne_df = pd.DataFrame(
            x_tsne,  columns=["composante_1", "composante_2"])
        x_tsne_df["clusters"] = cluster_labels

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silh_values = sample_silh_values[cluster_labels == i]

            ith_cluster_silh_values.sort()

            size_cluster_i = ith_cluster_silh_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0,
                              ith_cluster_silh_values,
                              facecolor=colors_2[i],
                              edgecolor=colors_2[i],
                              alpha=0.7)

            # Label the silhouette plots with
            # their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_score_avg[-1], color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        dx = fig.add_subplot(122)

        for i in range(n_clusters):
            dx.scatter(x_tsne_df[x_tsne_df.clusters == i]["composante_1"],
                       x_tsne_df[x_tsne_df.clusters == i]["composante_2"],
                       c=colors_2[i],
                       label='Cluster ' + str(i+1),
                       s=50)

        # Titres des axes
        dx.set_xlabel("Composante_1")
        dx.set_ylabel("Composante_2")
        dx.set_title("Projection 2D par TSNE")

        plt.suptitle(("Silhouette analysis for k-means clustering "
                      "clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()

        print("Silhouette score :", round(silhouette_score_avg[-1], 2), "\n")

    # Inertie, Silhouette_score, time_fit_and_predict
    return silhouette_score_avg
    
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

def calculate_ari(scaled_df, n_clusters, num_repetitions):
    # Stocker les labels de chaque run ici
    all_labels = []

    for i in range(num_repetitions):
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++")
        labels = clusterer.fit_predict(scaled_df)
        all_labels.append(labels)


    ari_values = []
    for i in range(num_repetitions):
        for j in range(i+1, num_repetitions):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_values.append(ari)

    # Calculer la moyenne et l'écart-type des valeurs d'ARI
    mean_ari = np.mean(ari_values)
    std_ari = np.std(ari_values)

    print(f"Valeurs ARI : {ari_values}")
    print(f"Moyenne ARI : {mean_ari}")
    print(f"Écart-type ARI : {std_ari}")

    return mean_ari, std_ari