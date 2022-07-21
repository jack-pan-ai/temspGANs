"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""
"""
Revised by Qilong Pan: 2022 4-7
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import torch
import os

def acf_visulization(ori_data, generated_data, save_name, epoch, args):
    data = {'Real': ori_data[1,0,:],
            'Generated': generated_data[1,0,:]}
    df = pd.DataFrame(data)
    fig = plt.figure()
    for i in range(df.shape[1]):
        pd.plotting.autocorrelation_plot(df.iloc[:, i], label = df.columns[i])
    plt.savefig(os.path.join(args.path_helper['log_path_img_pca'], f'{save_name}_epoch_{epoch + 1}_acfplot.png'),
                format="png")

def visualization (ori_data, generated_data, analysis, save_name, epoch, args, mean, cov):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([args.eval_num, len(ori_data), len(generated_data), 500])

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    if type(generated_data) is not np.ndarray:
        if type(generated_data) is torch.Tensor:
            generated_data = np.asarray(generated_data.to('cpu'))
    elif type(generated_data) is np.ndarray:
        generated_data = np.asarray(generated_data)
    else:
        print('The generated is not either torch.tensor, nor np.ndarray. Please check your data type for the output of generator.')
        raise TypeError

    no, channles, seq_len = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 0), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],0), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],0), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],0), [1,seq_len])))
    
    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        ## fixed x,y axis
        x_min = np.min(pca_results[:, 0]) * args.swell_ratio
        x_max = np.max(pca_results[:, 0]) * args.swell_ratio
        y_min = np.min(pca_results[:, 1]) * args.swell_ratio
        y_max = np.max(pca_results[:, 1]) * args.swell_ratio
        f, ax = plt.subplots(1)
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend(loc = 1)
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
#         plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)
        x_min = np.min(tsne_results[:anal_sample_no, 0]) * args.swell_ratio
        x_max = np.max(tsne_results[:anal_sample_no, 0]) * args.swell_ratio
        y_min = np.min(tsne_results[:anal_sample_no, 1]) * args.swell_ratio
        y_max = np.max(tsne_results[:anal_sample_no, 1]) * args.swell_ratio
        # Plotting
        f, ax = plt.subplots(1)
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
        ax.legend(loc=1)
        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
#         plt.show()

    plt.savefig(os.path.join(args.path_helper['log_path_img_pca'],f'{save_name}_epoch_{epoch+1}.png'), format="png")

    # mean difference
    gen_data_mean = np.sum(generated_data, 0)
    gen_data_diff = gen_data_mean - mean.reshape(gen_data_mean.shape)
    fig = plt.figure()
    plt.imshow(gen_data_diff, cmap='hot')
    plt.set_cmap('PiYG')
    plt.colorbar()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Difference of true mean and estimate mean from generated samples')
    plt.savefig(os.path.join(args.path_helper['log_path_img_pca'], f'{save_name}_epoch_{epoch + 1}_diffmean.png'),
                format="png")
    
    # marginal distribution
    fig = plt.figure()
    plt.hist(generated_data[:,0,0], 50 , density=True, facecolor='g', alpha=0.75)
    x = np.linspace(-3, 3, 50)
    y = np.exp(-(x) ** 2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(x, y, "r", linewidth=2)
    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.title('Marginal distribution at (0, 0)')
    plt.savefig(os.path.join(args.path_helper['log_path_img_pca'], f'{save_name}_epoch_{epoch + 1}_marginal.png'),
                format="png")

    # semigram
    
    

    # 1D line or 2D grid plot
    ## generated_data : [batch_size, channels, simu_dim]
    if channles ==1:
        fig = plt.figure()
        for i in range(args.num_lines):
            plt.plot(ori_data[i, :, :], color='orangered')
        plt.xlabel('Random Fields')
        plt.ylabel('Values')
        plt.title('Real Samples')
        plt.savefig(os.path.join(args.path_helper['log_path_img_pca'], f'{save_name}_epoch_{epoch + 1}_realline.png'), format="png")
        fig = plt.figure()
        for i in range(args.num_lines):
            plt.plot(generated_data[i, 0, :], color='lime')
        plt.xlabel('Random Fields')
        plt.ylabel('Values')
        plt.title('Generated Samples')
        plt.savefig(os.path.join(args.path_helper['log_path_img_pca'], f'{save_name}_epoch_{epoch + 1}_generatedline.png'),
                    format="png")
    else:
        # real
        fig = plt.figure()
        plt.imshow(ori_data[np.random.randint(no), :, :], cmap='hot')
        plt.set_cmap('PiYG')
        plt.colorbar()
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Heatmap for real 2D GRF')
        plt.savefig(os.path.join(args.path_helper['log_path_img_pca'], f'{save_name}_epoch_{epoch + 1}_realheatmap.png'),
                    format="png")

        # generated
        fig = plt.figure()
        plt.imshow(generated_data[np.random.randint(no), :, :], cmap='hot')
        plt.set_cmap('PiYG')
        plt.colorbar()
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Heatmap for generated 2D GRF')
        plt.savefig(os.path.join(args.path_helper['log_path_img_pca'], f'{save_name}_epoch_{epoch + 1}_generatedheatmap.png'),
                    format="png")

#    plt.show()