import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def visualization(config=None, datadict=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    element_centroids = datadict['raw']['x_test'][0,:,:]

    m = element_centroids[:,0]
    n = element_centroids[:,1]
    
    # 创建网格
    mi = np.linspace(np.min(m), np.max(m), 100)
    ni = np.linspace(np.min(n), np.max(n), 100)
    M, N = np.meshgrid(mi, ni)

    y_pred = datadict['test_results']['predictions']
    y_true = datadict['test_results']['ground_truth']
    metrics = datadict['test_results']['metrics']
    num_samples = y_true.shape[0]

    def handle_key_press(event):
        if event.key == ' ':
            for ax in axes.flat:
                ax.cla()

            idx = np.random.randint(0, num_samples)

            U1_PRED = griddata((m, n), y_pred[idx, :, 0].ravel(), (M, N), method='linear')
            U1_TRUE = griddata((m, n), y_true[idx, :, 0].ravel(), (M, N), method='linear')
            U2_PRED = griddata((m, n), y_pred[idx, :, 1].ravel(), (M, N), method='linear')
            U2_TRUE = griddata((m, n), y_true[idx, :, 1].ravel(), (M, N), method='linear')

            vmin1 = np.min((np.min(y_pred[idx, :, 0]), np.min(y_true[idx, :, 0])))
            vmax1 = np.max((np.max(y_pred[idx, :, 0]), np.max(y_true[idx, :, 0])))
            vmin2 = np.min((np.min(y_pred[idx, :, 1]), np.min(y_true[idx, :, 1])))
            vmax2 = np.max((np.max(y_pred[idx, :, 1]), np.max(y_true[idx, :, 1])))

            contour1 = axes[0, 0].contourf(M, N, U1_TRUE, cmap='viridis', vmin=vmin1, vmax=vmax1)
            contour2 = axes[0, 1].contourf(M, N, U1_PRED, cmap='viridis', vmin=vmin1, vmax=vmax1)
            contour3 = axes[1, 0].contourf(M, N, U2_TRUE, cmap='viridis', vmin=vmin2, vmax=vmax2)
            contour4 = axes[1, 1].contourf(M, N, U2_PRED, cmap='viridis', vmin=vmin2, vmax=vmax2)

            axes[0,0].set_title('{} True'.format(config['output_feature']))
            axes[0,1].set_title('{} Pred'.format(config['output_feature']))

            axes[0,1].text(0.95, 0.95, 'MAPE: {:.4f}'.format(metrics['MAPE'][idx]), fontsize=8,
                            transform=axes[0, 1].transAxes, horizontalalignment='right', verticalalignment='top')

            # TODO colorbar 没法清除
            """
            # 创建一个 colorbar，确保它适用于两个图
            cbar = fig.colorbar(contour1, ax=[ax[0], ax[1]], orientation='vertical')
            cbar.set_label('Colorbar Label')
            """

            # 添加文字标注
            axes[0,0].text(
                0.95, 0.95, f"Sample: {idx + 1}/{num_samples}",
                transform=axes[0,0].transAxes,
                horizontalalignment='right', verticalalignment='top'
            )
            # 给图形添加一个标题和x轴、y轴的标签
            plt.suptitle('Random Visualization')

            fig.canvas.draw()
        elif event.key == 'escape':
            # 如果按下了esc键，关闭图形并退出程序
            plt.close(fig)

    # 将按键事件绑定到图形上
    fig.canvas.mpl_connect('key_press_event', handle_key_press)

    # 显示图形并等待事件
    plt.show()
