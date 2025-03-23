import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if (os.path.exists(".\\data\\datasetClean.csv")):
        dataf: pd.DataFrame = pd.read_csv(".\\data\\datasetClean.csv")
    else :
        print("First run loadData.py to load the dataset.")
        exit()

    # Boxplot of Age at enrollment
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataf, x="Target", y="Age at enrollment")
    plt.title("Boxplot of Age at enrollment")
    plt.ylabel("Age at enrollment")
    plt.xlabel("Target")
    plt.savefig(".\\results\\boxplot_age.png")

    # Boxplot of Curricular units 1st sem (grade)
    plt.figure(figsize=(10, 6))
    sns.catplot(x="Target", y="Curricular units 1st sem (grade)", data=dataf,kind="box")
    plt.title("Boxplot: Grades in 1st Semester by Target")
    plt.xlabel("Target")
    plt.ylabel("Grades in 1st Semester")
    plt.savefig(".\\results\\boxplot_grade_1st_sem.png")
