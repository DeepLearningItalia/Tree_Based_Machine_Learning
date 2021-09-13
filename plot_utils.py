import plotly.graph_objs as go
import plotly
import plotly.express as px
plotly.offline.init_notebook_mode(connected = True)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


import numpy as np 
import pandas as pd

def _3d_plot(df_train, y_train, df_test,y_test, ml_model,plot_title="", train_pool=None, test_pool=None):
    
    column_names = list(df_train.columns)
    col_ranges = pd.concat([df_train.max(), df_train.min()],axis=1)
    col_ranges.columns=["max","min"]


    x_ml = np.linspace(start=col_ranges.loc[column_names[0],"min"],stop=col_ranges.loc[column_names[0],"max"],num=1000)
    y_ml = np.linspace(start=col_ranges.loc[column_names[1],"min"],stop=col_ranges.loc[column_names[1],"max"],num=1000)
    x_plot,y_plot = np.meshgrid(x_ml,y_ml)

    plt_data = np.array([x_plot, y_plot]).reshape(2, -1).T
    ml_df = pd.DataFrame(plt_data,columns=column_names)

    z = ml_model.predict_proba(ml_df)[:,1]
    z = z.reshape(len(y_ml),-1)

    ml_fig = go.Figure(data=[go.Surface(x=x_ml,
                                        y=y_ml,
                                        z=z,
                                        opacity=0.4)])
    ml_fig.update_layout(title=plot_title,
                         scene=dict(
                             xaxis_title=column_names[0],
                             yaxis_title=column_names[1],
                             zaxis_title='ML prediction')
                         )
    trace_points_train = go.Scatter3d(x=df_train[column_names[0]],
                                y=df_train[column_names[1]],
                                z=y_train,
                                mode='markers',
                                # name="training points",
                                # showlegend = False,
                                marker=dict(size=6,
                                            color="blue",
                                            opacity=0.8
                                            ))
    
    trace_points_test = go.Scatter3d(x=df_test[column_names[0]],
                            y=df_test[column_names[1]],
                            z=y_test,
                            mode='markers',
                            # name="test points",
                            # showlegend = False,
                            marker=dict(size=6,
                                        color="green",
                                        opacity=0.8
                                        ))

    ml_fig = ml_fig.add_trace(trace_points_train)
    ml_fig = ml_fig.add_trace(trace_points_test)

    ml_fig.show()


def classification_metric_plot(leaves,training_metrics, test_metrics,metric,plot_type=""):

    leaves_roc_plot = go.Figure(data=[go.Scatter(x=leaves,
                                        y=training_metrics,
                                       mode='lines+markers',
                                name="training "+metric,
                                  marker=dict(size=6,
                                            color="green",
                                            opacity=0.8
                                            ),
                                # showlegend = False,
                                        opacity=0.4)])
    leaves_roc_plot.update_layout(title= metric +" values for different "+ plot_type,
                         scene=dict(
                             xaxis_title="number of leaves",
                             yaxis_title=metric+" values")
                         )
    auc_test = go.Scatter(x=leaves,
                                        y=test_metrics,
                                       mode='lines+markers',
                                name="test "+metric,
                                # showlegend = False,
                                        opacity=0.4,
                                marker=dict(size=6,
                                            color="blue",
                                            opacity=0.8
                                            ))
    
    leaves_roc_plot = leaves_roc_plot.add_trace(auc_test)

    leaves_roc_plot.show()    
    
def roc_plot(target, predicted, show=True):
    """ROC curve for the model"""

    fpr, tpr, _ = roc_curve(target, predicted)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if show:
        plt.show()
    return roc_auc