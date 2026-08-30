# Turns the CSVs produced by run_experiments.py into PNG figures under results/figures/.
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_weighting_study(path, out_dir):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    c0_df = df[df.sweep == "c0"].sort_values("c0")
    fig, ax1 = plt.subplots()
    ax1.plot(c0_df["c0"], c0_df["HR@10"], "o-", label="HR@10")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("c0 (alpha=0)")
    ax1.set_ylabel("HR@10")
    ax2 = ax1.twinx()
    ax2.plot(c0_df["c0"], c0_df["NDCG@10"], "s--", color="orange", label="NDCG@10")
    ax2.set_ylabel("NDCG@10")
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.88))
    plt.title("Impact of c0 (missing-data weight) on eALS")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/weighting_c0.png", dpi=150)
    plt.close()

    a_df = df[df.sweep == "alpha"].sort_values("alpha")
    fig, ax1 = plt.subplots()
    ax1.plot(a_df["alpha"], a_df["HR@10"], "o-", label="HR@10")
    ax1.set_xlabel("alpha (popularity exponent)")
    ax1.set_ylabel("HR@10")
    ax2 = ax1.twinx()
    ax2.plot(a_df["alpha"], a_df["NDCG@10"], "s--", color="orange", label="NDCG@10")
    ax2.set_ylabel("NDCG@10")
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.88))
    plt.title("Impact of alpha (popularity exponent) on eALS")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/weighting_alpha.png", dpi=150)
    plt.close()


def plot_method_comparison(path, out_dir):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    for metric in ["HR@10", "NDCG@10"]:
        plt.figure()
        for name, g in df.groupby("method"):
            g = g.sort_values("iter")
            plt.plot(g["iter"], g[metric], marker="o", label=name)
        plt.xlabel("iteration")
        plt.ylabel(metric)
        plt.title(f"Convergence: {metric} vs. iteration")
        plt.legend()
        plt.tight_layout()
        fname = metric.replace("@", "_at_")
        plt.savefig(f"{out_dir}/convergence_{fname}.png", dpi=150)
        plt.close()


def plot_scalability(path, out_dir):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    k_df = df[df.axis == "K"].sort_values("K")
    if len(k_df):
        plt.figure()
        plt.plot(k_df["K"], k_df["seconds_per_iter"], "o-")
        plt.xlabel("K (latent factors)")
        plt.ylabel("seconds / iteration")
        plt.title("eALS training time per iteration vs. K")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/scalability_K.png", dpi=150)
        plt.close()

    p_df = df[df.axis == "n_partitions"].sort_values("n_partitions")
    if len(p_df):
        plt.figure()
        plt.plot(p_df["n_partitions"], p_df["seconds_per_iter"], "o-")
        plt.xlabel("Spark partitions")
        plt.ylabel("seconds / iteration")
        plt.title("eALS training time per iteration vs. parallelism")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/scalability_partitions.png", dpi=150)
        plt.close()

    d_df = df[df.axis == "data_frac"].sort_values("n_train")
    if len(d_df):
        plt.figure()
        plt.plot(d_df["n_train"], d_df["seconds_per_iter"], "o-")
        plt.xlabel("training interactions")
        plt.ylabel("seconds / iteration")
        plt.title("eALS training time per iteration vs. data size")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/scalability_data_size.png", dpi=150)
        plt.close()


def plot_mllib_reference(path, out_dir):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path).sort_values("K")
    plt.figure()
    plt.plot(df["K"], df["seconds_per_iter"], "o-")
    plt.xlabel("K (latent factors)")
    plt.ylabel("seconds / iteration")
    plt.title("Spark MLlib ALS (Hu et al.) training time per iteration vs. K")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mllib_scalability_K.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    out_dir = f"{args.results}/figures"
    os.makedirs(out_dir, exist_ok=True)

    plot_weighting_study(f"{args.results}/weighting_study.csv", out_dir)
    plot_method_comparison(f"{args.results}/method_comparison.csv", out_dir)
    plot_scalability(f"{args.results}/scalability_study.csv", out_dir)
    plot_mllib_reference(f"{args.results}/spark_mllib_als.csv", out_dir)
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
