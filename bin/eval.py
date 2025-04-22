# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

import src.utils as ut
import src.fuzz as fz
import src.classif as cl

def main():
    """
    Full KNN Fuzz version with Choquet integral Leave-One-Out evaluation.
    """
    # Hyperparameters
    dim = 4
    nb_classes = 3
    nb_points_per_class = 100
    
    # Generate positive gaussian random variables
    data, labels = ut.dynamic_generate_positive_gaussian_data(
        dim=dim,
        nb_classes=nb_classes,
        nb_points_per_class=nb_points_per_class,
        seed=42
    )
    print(f"Generated {data.shape[0]} all-positive samples in {dim}D space.")

    # Normalize data
    data_norm = fz.batch_norm(data)
    data_labels = ut.convert_to_float_lst(labels)
    print(f"Normalized data: {data_norm.shape[0]} samples in {data_norm.shape[1]}D space.")

    # LOO evaluation
    l = []
    k = 10

    tic = time.time()
    for k in range(1, k+1): 
        print(f"Leave one out avec k = {k}")
        
        res = cl.leave_one_out(
                C = cl.KNNFuzz(input_dimension=data_norm.shape[1], k=k, sim=fz.SimLevel1), 
                DS = (data_norm, data_labels)
            )
        l.append(res)

    toc = time.time()
    print(f"Result in {(toc-tic):0.4f} seconds.")
    print(f"Result: {l}")
    plt.plot(l, label="Accuracy en fonction de k")
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.grid()
    plt.legend()
    plt.savefig("archive/accuracy.png")


if __name__ == "__main__":
    main()
    print("Execution completed.")

