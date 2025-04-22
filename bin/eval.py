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
    # Define list for each sim level
    l1 = []
    l2 = []
    l3 = []
    k = 10

    for i in range(10):
        print("Epoch ", i)
        mu = fz.generate_capacity(ut.enumerate_permute_batch(data_norm[0]),2**len(data_norm[0])-1)

        tic = time.time()
        for k in range(1, k+1): 
            print(f"Leave one out avec k = {k}")
            
            # SimLevel1
            res = cl.leave_one_out(
                    C = cl.KNNFuzz(input_dimension=data_norm.shape[1],mu=mu, k=k, sim=fz.SimLevel1), 
                    DS = (data_norm, data_labels)
                )
            l1.append(res)

            # Sim level 2
            res = cl.leave_one_out(
                    C = cl.KNNFuzz(input_dimension=data_norm.shape[1],mu=mu, k=k, sim=fz.SimLevel2), 
                    DS = (data_norm, data_labels)
                )
            l2.append(res)

            # Sim level 3
            res = cl.leave_one_out(
                    C = cl.KNNFuzz(input_dimension=data_norm.shape[1],mu=mu, k=k, sim=fz.SimLevel3), 
                    DS = (data_norm, data_labels)
                )
            l3.append(res)

        toc = time.time()
        print(f"Result in {(toc-tic):0.4f} seconds.")

    # Plot results
    plt.plot(l1, label="Sim level 1")
    plt.plot(l2, label="Sim level 2")
    plt.plot(l3, label="Sim level 3")

    # Config additional params
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.grid()
    plt.legend()
    plt.savefig("archive/accuracy.png")


    print(f"Result in {(toc-tic):0.4f} seconds.")

    # Plot results
    plt.plot(l1, label="Sim level 1")
    plt.plot(l2, label="Sim level 2")
    plt.plot(l3, label="Sim level 3")

    # Config additional params
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.grid()
    plt.legend()
    plt.savefig("archive/accuracy.png")


if __name__ == "__main__":
    main()
    print("Execution completed.")

