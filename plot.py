import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from yellowbrick.cluster import KElbowVisualizer

from error import error_handler


class Plot(object):
    @staticmethod
    @error_handler
    def create_bar_chart_from_crosstab_of_column(df, columns, tilt=False):
        for column in columns:
            ct = pd.crosstab(df[column], df.readmitted)
            st = ct.stack().reset_index().rename(columns={0: 'count'})
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=column, y="count", hue='readmitted', data=st).set(
                title='The total amount of readmitted count for each ' + column)

            if tilt:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            for p in ax.patches:
                ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 7), textcoords='offset points')

            plt.show()

            if not column.startswith('diag_'):
                sns.lineplot(data=st, x=column, y="count", hue="readmitted").set(
                    title='The change in readmitted count for each ' + column)

                plt.show()

    @staticmethod
    @error_handler
    def create_conditional_histograms(df, columns, target_column):
        for column in columns:
            g = sns.FacetGrid(df, col=target_column, margin_titles=True, aspect=1)
            g.map(plt.hist, column)

            for ax in g.axes.flat:
                labels = ax.get_xticklabels()
                ax.set_xticklabels(labels, rotation=40)

    # Creates a Confusion Matrix for a specified model using a Confusion Matrix Display
    @staticmethod
    @error_handler
    def create_confusion_matrix(model, test_y, predicted):
        cm = confusion_matrix(test_y, predicted, labels=model.classes_)
        graph = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        graph.plot(values_format='')
        graph.ax_.set_title(str(model))
        plt.show()

    @staticmethod
    @error_handler
    def elbow(df, meth='distortion'):
        # Selects the KMeans model to find to the amount of clusters needed
        model = KMeans()
        # Creates the Graph using a specified method to determine an amout of clusters needed
        visualizer = KElbowVisualizer(
            model, k=(4, 12), metric=meth, timings=True, locate_elbow=True
        )
        visualizer.fit(df)
        visualizer.show()

    @staticmethod
    @error_handler
    def cluster_graphs(df):
        pca = PCA(2)
        df_clust = pca.fit_transform(df)
        # uses a Kmeans module
        model = KMeans(n_clusters=6, random_state=0)
        # Predicts cluster point positions
        label = model.fit_predict(df_clust)
        # Find all the unique labels
        unique_labels = np.unique(label)
        # Finds all Kmeans cluster centers
        centroids = model.cluster_centers_

        # Draws a scatter plot of points using unique labels and places centroids
        for i in unique_labels:
            plt.scatter(df_clust[label == i, 0], df_clust[label == i, 1], label=i)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
        plt.title('Kmeans Clusters')
        plt.legend()
        plt.show()

        # Grouping and Exporting Clusters to CSV
        md = pd.Series(label)
        df['clust'] = md
        df.groupby('clust').mean().to_csv('grouped_clusters.csv')

        # Plots a Histogram of Clusters
        plt.hist(df['clust'], color='skyblue', ec='black')
        plt.title('Cluster Frequency')
        plt.xlabel('Cluster')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()

    @staticmethod
    @error_handler
    def create_coefficient_chart(df, coefficients):
        columns = df.columns.tolist()
        coefficients = coefficients[0]
        columns.remove("readmitted")
        coefficients = sorted(zip(columns, coefficients), key=lambda coefficient_tuple: coefficient_tuple[1])
        coefficients_positive = [coefficient_tuple for coefficient_tuple in coefficients if coefficient_tuple[1] >= 0]
        coefficients_negative = [coefficient_tuple for coefficient_tuple in coefficients if coefficient_tuple[1] < 0]
        positive = plt.figure()
        plt.barh(*zip(*coefficients_positive), figure=positive)
        plt.title("Positive Logistic Regression Coefficients", figure=positive)
        plt.ylabel("Columns", figure=positive)
        plt.xlabel("Weight", figure=positive)
        negative = plt.figure()
        plt.barh(*zip(*coefficients_negative), figure=negative)
        plt.title("Negative Logistic Regression Coefficients", figure=negative)
        plt.ylabel("Columns", figure=negative)
        plt.xlabel("Weight", figure=negative)
        plt.subplots_adjust(left=0.19)
        plt.show()

    @staticmethod
    @error_handler
    def show_plots():
        plt.tight_layout()
        plt.show()
