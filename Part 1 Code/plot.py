import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class Plot(object):
    @staticmethod
    def create_bar_chart_from_crosstab_of_column(df, columns, tilt=False):
        for column in columns:
            ct = pd.crosstab(df[column], df.readmitted)
            st = ct.stack().reset_index().rename(columns={0: 'count'})
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=column, y="count", hue='readmitted', data=st).set(title="The total amount of readmitted count for each " + column)

            if tilt:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            for p in ax.patches:
                ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 7), textcoords='offset points')

            plt.show()

            if(not column.startswith('diag_') and not column.startswith('race')):
                sns.lineplot(data=st, x=column, y="count", hue="readmitted").set(title="The change in readmitted count for each " + column)

                plt.show()

    @staticmethod
    def create_conditional_histograms(df, columns, target_column):
        for column in columns:
            g = sns.FacetGrid(df, col=target_column, margin_titles=True, aspect=1)
            g.map(plt.hist, column)

            for ax in g.axes.flat:
                labels = ax.get_xticklabels()
                ax.set_xticklabels(labels, rotation=40)

    @staticmethod
    def create_confusion_matrix(model, test_y, predicted):
        cm = confusion_matrix(test_y, predicted, labels=model.classes_)
        graph = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        graph.plot(values_format='')
        graph.ax_.set_title(str(model))

    @staticmethod
    def show_plots():
        plt.show()


